import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np
from pytorchBaselines.a2c_ppo_acktr.utils import init


class RNNBase(nn.Module):
    # edge: True -> edge RNN, False -> node RNN
    def __init__(self, config, edge):
        super(RNNBase, self).__init__()
        self.config = config

        # if this is an edge RNN
        if edge:
            self.gru = nn.GRU(config.SRNN.human_human_edge_embedding_size, config.SRNN.human_human_edge_rnn_size)
        # if this is a node RNN
        else:
            self.gru = nn.GRU(config.SRNN.human_node_embedding_size*2, config.SRNN.human_node_rnn_size)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    # x: [seq_len, nenv, 6 or 30 or 36, ?]
    # hxs: [1, nenv, 6 or 30 or 36, ?]
    # masks: [1, nenv, 1]
    def _forward_gru(self, x, hxs, masks):
        # for acting model, input shape[0] == hidden state shape[0]
        if x.size(0) == hxs.size(0):
            # use env dimension as batch
            # [1, 12, 6, ?] -> [1, 12*6, ?] or [30, 6, 6, ?] -> [30, 6*6, ?]
            seq_len, nenv, agent_num, _ = x.size()
            x = x.view(seq_len, nenv*agent_num, -1)
            hxs_times_masks = hxs * (masks.view(seq_len, nenv, 1, 1))
            hxs_times_masks = hxs_times_masks.view(seq_len, nenv*agent_num, -1)
            x, hxs = self.gru(x, hxs_times_masks) # we already unsqueezed the inputs in SRNN forward function
            x = x.view(seq_len, nenv, agent_num, -1)
            hxs = hxs.view(seq_len, nenv, agent_num, -1)

        # during update, input shape[0] * nsteps (30) = hidden state shape[0]
        else:

            # N: nenv, T: seq_len, agent_num: node num or edge num
            T, N, agent_num, _ = x.size()
            # x = x.view(T, N, agent_num, x.size(2))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            # for the [29, num_env] boolean array, if any entry in the second axis (num_env) is True -> True
            # to make it [29, 1], then select the indices of True entries
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            # hxs = hxs.unsqueeze(0)
            # hxs = hxs.view(hxs.size(0), hxs.size(1)*hxs.size(2), hxs.size(3))
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                # x and hxs have 4 dimensions, merge the 2nd and 3rd dimension
                x_in = x[start_idx:end_idx]
                x_in = x_in.view(x_in.size(0), x_in.size(1)*x_in.size(2), x_in.size(3))
                hxs = hxs.view(hxs.size(0), N, agent_num, -1)
                hxs = hxs * (masks[start_idx].view(1, -1, 1, 1))
                hxs = hxs.view(hxs.size(0), hxs.size(1) * hxs.size(2), hxs.size(3))
                rnn_scores, hxs = self.gru(x_in, hxs)

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T, N, agent_num, -1)
            hxs = hxs.view(1, N, agent_num, -1)

        return x, hxs


class HumanNodeRNN(RNNBase):
    '''
    Class representing human Node RNNs in the st-graph
    '''
    def __init__(self, config):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(HumanNodeRNN, self).__init__(config, edge=False)

        self.config = config

        # Store required sizes
        self.rnn_size = config.SRNN.human_node_rnn_size
        self.output_size = config.SRNN.human_node_output_size
        self.embedding_size = config.SRNN.human_node_embedding_size
        self.input_size = config.SRNN.human_node_input_size
        self.edge_rnn_size = config.SRNN.human_human_edge_rnn_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()


        # Linear layer to embed edgeRNN hidden states
        self.edge_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)

        # Linear layer to embed attention module output
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size*2, self.embedding_size)


        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)



    def forward(self, pos, h_temporal, h_spatial_other, h, masks):
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(pos)
        encoded_input = self.relu(encoded_input)

        # Concat both the embeddings
        h_edges = torch.cat((h_temporal, h_spatial_other), -1)
        h_edges_embedded = self.relu(self.edge_attention_embed(h_edges))

        concat_encoded = torch.cat((encoded_input, h_edges_embedded), -1)

        x, h_new = self._forward_gru(concat_encoded, h, masks)

        outputs = self.output_linear(x)


        return outputs, h_new


class HumanHumanEdgeRNN(RNNBase):
    '''
    Class representing the Human-Human Edge RNN in the s-t graph
    '''
    def __init__(self, config):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(HumanHumanEdgeRNN, self).__init__(config, edge=True)

        self.config = config

        # Store required sizes
        self.rnn_size = config.SRNN.human_human_edge_rnn_size
        self.embedding_size = config.SRNN.human_human_edge_embedding_size
        self.input_size = config.SRNN.human_human_edge_input_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)
        self.relu = nn.ReLU()


    def forward(self, inp, h, masks):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        h : hidden state of the current edgeRNN
        c : cell state of the current edgeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(inp)
        encoded_input = self.relu(encoded_input)

        x, h_new = self._forward_gru(encoded_input, h, masks)

        return x, h_new


class EdgeAttention(nn.Module):
    '''
    Class representing the attention module
    '''
    def __init__(self, config):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EdgeAttention, self).__init__()

        self.config = config

        # Store required sizes
        self.human_human_edge_rnn_size = config.SRNN.human_human_edge_rnn_size
        self.human_node_rnn_size = config.SRNN.human_node_rnn_size
        self.attention_size = config.SRNN.attention_size



        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer=nn.ModuleList()
        self.spatial_edge_layer=nn.ModuleList()

        self.temporal_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))

        # Linear layer to embed spatial edgeRNN hidden states
        self.spatial_edge_layer.append(nn.Linear(self.human_human_edge_rnn_size, self.attention_size))



        # number of agents who have spatial edges (complete graph: all 6 agents; incomplete graph: only the robot)
        self.agent_num = 1
        self.num_attention_head = 1

    def att_func(self, temporal_embed, spatial_embed, h_spatials):
        seq_len, nenv, num_edges, h_size = h_spatials.size()  # [1, 12, 30, 256] in testing,  [12, 30, 256] in training
        attn = temporal_embed * spatial_embed
        attn = torch.sum(attn, dim=3)

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)

        # Softmax

        attn = attn.view(seq_len, nenv, self.agent_num, self.human_num)
        attn = torch.nn.functional.softmax(attn, dim=-1)

        # Compute weighted value
        # weighted_value = torch.mv(torch.t(h_spatials), attn)

        # reshape h_spatials and attn
        # shape[0] = seq_len, shape[1] = num of spatial edges (6*5 = 30), shape[2] = 256
        h_spatials = h_spatials.view(seq_len, nenv, self.agent_num, self.human_num, h_size)
        h_spatials = h_spatials.view(seq_len * nenv * self.agent_num, self.human_num, h_size).permute(0, 2,
                                                                                         1)  # [seq_len*nenv*6, 5, 256] -> [seq_len*nenv*6, 256, 5]

        attn = attn.view(seq_len * nenv * self.agent_num, self.human_num).unsqueeze(-1)  # [seq_len*nenv*6, 5, 1]
        weighted_value = torch.bmm(h_spatials, attn)  # [seq_len*nenv*6, 256, 1]

        # reshape back
        weighted_value = weighted_value.squeeze(-1).view(seq_len, nenv, self.agent_num, h_size)  # [seq_len, 12, 6 or 1, 256]
        return weighted_value, attn



    # h_temporal: [seq_len, nenv, 1, 256]
    # h_spatials: [seq_len, nenv, 5, 256]
    def forward(self, h_temporal, h_spatials):
        '''
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edgeRNN
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node.
        '''
        # find the number of humans by the size of spatial edgeRNN hidden state
        self.human_num = h_spatials.size()[2] // self.agent_num

        weighted_value_list, attn_list=[],[]
        for i in range(self.num_attention_head):

            # Embed the temporal edgeRNN hidden state
            temporal_embed = self.temporal_edge_layer[i](h_temporal)
            # temporal_embed = temporal_embed.squeeze(0)

            # Embed the spatial edgeRNN hidden states
            spatial_embed = self.spatial_edge_layer[i](h_spatials)

            # Dot based attention
            try:
                temporal_embed = temporal_embed.repeat_interleave(self.human_num, dim=2)
            except RuntimeError:
                print('hello')
            weighted_value,attn=self.att_func(temporal_embed, spatial_embed, h_spatials)
            weighted_value_list.append(weighted_value)
            attn_list.append(attn)

        if self.num_attention_head > 1:
            return self.final_attn_linear(torch.cat(weighted_value_list, dim=-1)), attn_list
        else:
            return weighted_value_list[0], attn_list[0]


class SRNN(nn.Module):
    """
    Class representing the SRNN model
    """
    def __init__(self, obs_space_dict, config, infer=False):
        """
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        """
        super(SRNN, self).__init__()
        self.infer = infer
        self.is_recurrent = True
        self.config=config

        self.human_num = config.sim.human_num

        self.seq_length = config.ppo.num_steps
        self.nenv = config.training.num_processes
        self.nminibatch = config.ppo.num_mini_batch

        # Store required sizes
        self.human_node_rnn_size = config.SRNN.human_node_rnn_size
        self.human_human_edge_rnn_size = config.SRNN.human_human_edge_rnn_size
        self.output_size = config.SRNN.human_node_output_size

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = HumanNodeRNN(config)
        self.humanhumanEdgeRNN_spatial = HumanHumanEdgeRNN(config)
        self.humanhumanEdgeRNN_temporal = HumanHumanEdgeRNN(config)

        # Initialize attention module
        self.attn = EdgeAttention(config)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        num_inputs = hidden_size = self.output_size

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())


        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.robot_linear = init_(nn.Linear(7, 3))
        self.human_node_final_linear=init_(nn.Linear(self.output_size,2))

        self.num_edges = self.human_num + 1 # number of spatial edges + number of temporal edges
        self.temporal_edges = [0]
        self.spatial_edges = np.arange(1, self.human_num+1)


    def forward(self, inputs, rnn_hxs, masks, infer=False):
        if infer:
            # Test time
            seq_length = 1
            nenv = self.nenv

        else:
            # Training time
            seq_length = self.seq_length
            nenv = self.nenv // self.nminibatch

        robot_node = reshapeT(inputs['robot_node'], seq_length, nenv)
        temporal_edges = reshapeT(inputs['temporal_edges'], seq_length, nenv)
        spatial_edges = reshapeT(inputs['spatial_edges'], seq_length, nenv)

        hidden_states_node_RNNs = reshapeT(rnn_hxs['human_node_rnn'], 1, nenv)
        hidden_states_edge_RNNs = reshapeT(rnn_hxs['human_human_edge_rnn'], 1, nenv)
        masks = reshapeT(masks, seq_length, nenv)

        if not self.config.training.cuda:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, self.num_edges, rnn_hxs['human_human_edge_rnn'].size()[-1]).cpu())
        else:
            all_hidden_states_edge_RNNs = Variable(
                torch.zeros(1, nenv, self.num_edges, rnn_hxs['human_human_edge_rnn'].size()[-1]).cuda())


        # Do forward pass through temporaledgeRNN
        hidden_temporal_start_end=hidden_states_edge_RNNs[:,:,self.temporal_edges,:]
        output_temporal, hidden_temporal = self.humanhumanEdgeRNN_temporal(temporal_edges, hidden_temporal_start_end, masks)

        # Update the hidden state and cell state
        all_hidden_states_edge_RNNs[:, :, self.temporal_edges,:] = hidden_temporal

        # Spatial Edges
        hidden_spatial_start_end=hidden_states_edge_RNNs[:,:,self.spatial_edges,:]
        # Do forward pass through spatialedgeRNN
        output_spatial, hidden_spatial = self.humanhumanEdgeRNN_spatial(spatial_edges, hidden_spatial_start_end, masks)

        # Update the hidden state and cell state
        all_hidden_states_edge_RNNs[:, :,self.spatial_edges,: ] = hidden_spatial


        # Do forward pass through attention module
        hidden_attn_weighted, _ = self.attn(output_temporal, output_spatial)


        # concatenate human node features with robot features
        nodes_current_selected = self.robot_linear(robot_node)

        # Do a forward pass through nodeRNN
        outputs, h_nodes \
            = self.humanNodeRNN(nodes_current_selected, output_temporal, hidden_attn_weighted, hidden_states_node_RNNs, masks)


        # Update the hidden and cell states
        all_hidden_states_node_RNNs = h_nodes
        outputs_return = outputs

        rnn_hxs['human_node_rnn'] = all_hidden_states_node_RNNs
        rnn_hxs['human_human_edge_rnn'] = all_hidden_states_edge_RNNs


        # x is the output of the robot node and will be sent to actor and critic
        x = outputs_return[:, :, 0, :]

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        for key in rnn_hxs:
            rnn_hxs[key] = rnn_hxs[key].squeeze(0)

        if infer:
            return self.critic_linear(hidden_critic).squeeze(0), hidden_actor.squeeze(0), rnn_hxs
        else:
            return self.critic_linear(hidden_critic).view(-1, 1), hidden_actor.view(-1, self.output_size), rnn_hxs


def reshapeT(T, seq_length, nenv):
    shape = T.size()[1:]
    return T.unsqueeze(0).reshape((seq_length, nenv, *shape))