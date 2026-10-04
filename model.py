"""
Message-Passing GNNs from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - edges_to_coo
import torch
def edges_to_coo(edge_list, num_nodes=None):
    # TODO: Convert a list of (src, dst) edge pairs into COO-format src/dst tensors.
    if isinstance(edge_list,list):
        if len(edge_list) == 0:
            return torch.tensor([],dtype = torch.long), torch.tensor([],dtype=torch.long), num_nodes if num_nodes is not None else 0
        edge_tensor = torch.tensor(edge_list,dtype = torch.long)
    else:
        edge_tensor = edge_list
    if edge_tensor.numel() == 0:
        return torch.tensor([],dtype = torch.long), torch.tensor([],dtype=torch.long), num_nodes
    src = edge_tensor[:,0]
    dst = edge_tensor[:,1]
    if num_nodes is None:
        num_nodes = int(torch.max(edge_tensor).item())+1
    return src,dst,num_nodes
    pass

# Step 2 - add_self_loops
import torch
def add_self_loops(src, dst, num_nodes):
    """Append self-loop edges (i, i) for every node to COO edge indices.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes in the graph.

    Returns:
        src_out: LongTensor [E + num_nodes]
        dst_out: LongTensor [E + num_nodes]
    """
    self_loop = torch.arange(num_nodes,dtype = src.dtype)
    src_out = torch.cat([src,self_loop])
    dst_out = torch.cat([dst,self_loop])
    return src_out, dst_out
    # TODO: Append self-loop edges (i, i) for every node to the COO tensors
    pass

# Step 3 - compute_node_degrees
def compute_node_degrees(src, dst, num_nodes, edge_weight=None):
    """Compute per-node in-degrees (optionally weighted) from COO edges.

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].

    Returns:
        FloatTensor: In-degrees of shape [N].
    """
    # TODO: Compute per-node in-degrees by scattering onto destination nodes
    deg = torch.zeros(num_nodes, dtype = torch.float)
    if edge_weight is None:
        weights = torch.ones(src.size(0),dtype = torch.float)
        
    else:
        weights = edge_weight
    deg.index_add_(0,dst, weights)
    return deg
    pass

# Step 4 - symmetric_normalize_edge_weights
def symmetric_normalize_edge_weights(src, dst, num_nodes, edge_weight=None):
    """Compute symmetrically normalized edge weights w_ij / sqrt(d_i * d_j).

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].
            Defaults to all ones (float32) when None.

    Returns:
        FloatTensor: Symmetrically normalized weights of shape [E].
    """
    # TODO: Compute symmetrically normalized edge weights for GCN-style propagation.
    deg = compute_node_degrees(src, dst, num_nodes, edge_weight)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg == 0] = 0.0
    
    if edge_weight is None:
        weights = torch.ones(src.size(0), dtype=torch.float)
    else:
        weights = edge_weight

    norm_weights = weights * deg_inv_sqrt[src] * deg_inv_sqrt[dst]
    return norm_weights
    pass

# Step 5 - gather_source_node_features
def gather_source_node_features(node_features, src):
    # TODO: Return edge-aligned source feature rows (E, F) from node_features.

    return node_features[src]
    pass

# Step 6 - scatter_sum_to_nodes
def scatter_sum_to_nodes(edge_features, dst, num_nodes):
    """Scatter-sum edge features onto destination nodes to produce per-node aggregated vectors.

    Args:
        edge_features: FloatTensor of shape (E, F) with one feature row per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (N, F); row j is the sum of edge features with dst == j.
    """
    # TODO: Scatter-sum edge features onto destination nodes to produce per-node vectors
    output = torch.zeros(num_nodes, edge_features.size(1), dtype=edge_features.dtype, device=edge_features.device)
    # Accumulate edge representations into target receiver nodes
    output.index_add_(0, dst, edge_features)
    return output

    pass

# Step 7 - scatter_mean_to_nodes
def scatter_mean_to_nodes(edge_features, dst, num_nodes):
    # TODO: Scatter-mean edge features onto destination nodes (sum then divide by in-degree).
    sum_features = scatter_sum_to_nodes(edge_features,dst,num_nodes)
    deg = torch.bincount(dst,minlength = num_nodes).float()
    deg_inv = deg.pow(-1.0)
    deg_inv[deg == 0] = 0.0
    mean_features = sum_features * deg_inv.reshape(-1,1)
    return mean_features

    pass

# Step 8 - scatter_max_to_nodes
def scatter_max_to_nodes(edge_features, dst, num_nodes):
    # TODO: Scatter-max edge features onto destination nodes (elementwise max).
    out = edge_features.new_full((num_nodes, edge_features.size(1)), float('-inf'))
    out.index_reduce_(0, dst, edge_features, reduce="amax")
    return out
    pass

# Step 9 - compute_messages
def compute_messages(node_features, src, dst, message_fn, edge_attr=None):
    """Build per-edge messages via gather + message_fn.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        messages: FloatTensor of shape (E, M).
    """
    # TODO: Build per-edge messages by gathering features and applying message_fn
    x_src = gather_source_node_features(node_features,src)
    x_dst = gather_source_node_features(node_features,dst)
    if edge_attr is not None:
        return message_fn(x_src,x_dst,edge_attr)
    else:
        return message_fn(x_src,x_dst)
    pass

# Step 10 - aggregate_messages
def aggregate_messages(messages, dst, num_nodes, aggr='sum'):
    """Aggregate edge messages onto destination nodes using sum, mean, or max.

    Args:
        messages: FloatTensor of shape (E, M) with one message vector per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.
        aggr: str in {'sum', 'mean', 'max'} selecting the reduction.

    Returns:
        FloatTensor of shape (N, M); row j is the aggregated message for node j.
    """
    # TODO: Aggregate edge messages onto destination nodes via sum/mean/max...
    if aggr == 'sum' : 
        return scatter_sum_to_nodes(messages,dst,num_nodes)
    elif aggr == 'mean':
        return scatter_mean_to_nodes(messages,dst,num_nodes)
    elif aggr == 'max':
        return scatter_max_to_nodes(messages,dst,num_nodes)
    else:
        raise ValueError(f"unknown aggregation mode: {aggr}")
    pass

# Step 11 - update_node_features
def update_node_features(node_features, aggregated, update_fn):
    # TODO: Implement update_node_features to fuse each node's current state with its aggregated...
    return update_fn(node_features,aggregated)
    pass

# Step 12 - message_passing_layer
def message_passing_layer(node_features, src, dst, message_fn, update_fn, aggr='sum', edge_attr=None):
    """Run one full Gilmer MPNN step: message, aggregate, and update.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages (E, M).
        update_fn: callable(node_features, aggregated) -> updated (N, H).
        aggr: str in {'sum', 'mean', 'max'}.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        updated_features: FloatTensor of shape (N, H).
    """
    num_nodes = node_features.size(0)
    # TODO: compose message, aggregate, and update into one MPNN step
    # 1. Compute messages for all edges
    messages = compute_messages(node_features, src, dst, message_fn, edge_attr=edge_attr)
    
    # 2. Aggregate messages at destination nodes
    aggregated = aggregate_messages(messages, dst, num_nodes, aggr=aggr)
    
    # 3. Update node features by fusing current state with aggregated messages
    updated_features = update_node_features(node_features, aggregated, update_fn)
    
    return updated_features
    pass

# Step 13 - stack_message_passing_layers
def stack_message_passing_layers(node_features, src, dst, layers, edge_attr=None):
    """Apply a sequence of message-passing layer callables to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        layers: list of callables, each
            layer(node_features, src, dst, edge_attr=None) -> Tensor (N, H_i).
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        embeddings: FloatTensor of shape (N, H), final layer output.
        all_layer_outputs: list of FloatTensors, one per layer (N, H_i).
    """
    # TODO: Apply a sequence of MP layer callables; return final + intermediates
    intermediate_outputs = [] #stack
    current_features = node_features
    for layer in layers:
        current_features = layer(current_features, src, dst, edge_attr=edge_attr)
        intermediate_outputs.append(current_features)
    return current_features, intermediate_outputs
    pass

# Step 14 - gcn_renormalize_adjacency
def gcn_renormalize_adjacency(src, dst, num_nodes):
    """Apply Kipf-Welling renormalization: self-loops then symmetric norm.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes N.

    Returns:
        src_hat: LongTensor [E + N] sources after self-loops.
        dst_hat: LongTensor [E + N] destinations after self-loops.
        norm_weight: FloatTensor [E + N] symmetrically normalized weights.
    """
    # TODO: add self-loops then symmetrically normalize the adjacency...
    src_loop, dst_loop = add_self_loops(src, dst, num_nodes)
    edge_weights = symmetric_normalize_edge_weights(src_loop,dst_loop,num_nodes)
    return src_loop, dst_loop,edge_weights
    pass

# Step 15 - gcn_linear_transform
import torch
def gcn_linear_transform(node_features, weight, bias=None):
    """Apply the GCN linear feature transform X @ W (+ bias).

    Args:
        node_features: FloatTensor of shape (N, Fin).
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout).

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # TODO: compute the matrix product and optionally add a bias vector
    out = torch.matmul(node_features, weight)
    if bias is not None:
        out = out+bias
    return out
    pass

# Step 16 - gcn_layer_forward
import torch

def gcn_layer_forward(node_features, src, dst, weight, bias=None, num_nodes=None, activation=None):
    """Forward pass of one GCN layer: renormalize, transform, propagate.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; defaults to node_features.shape[0].
        activation: optional callable applied to the output.

    Returns:
        FloatTensor of shape (N, Fout).
    """
    # 1. Default num_nodes if not provided
    if num_nodes is None:
        num_nodes = node_features.shape[0]
        
    # 2. Apply GCN renormalization (adds self-loops and computes symmetric normalization weights)
    src_loop, dst_loop, edge_weights = gcn_renormalize_adjacency(src, dst, num_nodes)
    
    # 3. Apply linear feature transformation (XW + b)
    transformed_features = gcn_linear_transform(node_features, weight, bias)
    
    # 4. Define message function and update function for propagation
    def gcn_message_fn(x_src, x_dst, weights):
        return x_src * weights.unsqueeze(-1)
        
    def gcn_update_fn(node_feats, aggregated):
        return aggregated
        
    # 5. Run message passing layer
    out = message_passing_layer(
        transformed_features,
        src_loop,
        dst_loop,
        message_fn=gcn_message_fn,
        update_fn=gcn_update_fn,
        aggr='sum',
        edge_attr=edge_weights
    )
    
    # 6. Apply optional activation function (e.g., ReLU)
    if activation is not None:
        out = activation(out)
        
    return out

# Step 17 - init_gcn_parameters
import torch

def init_gcn_parameters(in_dim, out_dim, with_bias=True, seed=None):
    """
    Initializes a GCN layer weight matrix and optional bias with Glorot uniform initialization.
    
    Args:
        in_dim (int): Input feature dimension (Fin).
        out_dim (int): Output feature dimension (Fout).
        with_bias (bool): Whether to include a bias vector.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        dict: Dictionary containing 'weight' and optionally 'bias' as float Tensors.
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    # Calculate Glorot uniform bound: a = sqrt(6 / (in_dim + out_dim))
    a = torch.sqrt(torch.tensor(6.0 / (in_dim + out_dim)))
    
    # Sample weight uniformly from [-a, a]
    weight = torch.empty(in_dim, out_dim).uniform_(-a.item(), a.item())
    
    params = {'weight': weight}
    
    if with_bias:
        bias = torch.zeros(out_dim)
        params['bias'] = bias
        
    return params

# Step 18 - gcn_stack_forward
import torch

def gcn_stack_forward(node_features, src, dst, param_list, activations=None, num_nodes=None):
    """
    Produces deep node embeddings from a stack of GCN layers.
    
    Args:
        node_features (Tensor): Initial node feature tensor of shape (N, F0).
        src (LongTensor): Source node indices of shape (E,).
        dst (LongTensor): Destination node indices of shape (E,).
        param_list (list of dict): Each dict contains 'weight' (Fin, Fout) 
                                   and optional 'bias' (Fout,).
        activations (list of callable or None, optional): Activation function per layer.
        num_nodes (int, optional): Total number of nodes N. Defaults to node_features.shape[0].
        
    Returns:
        tuple: (embeddings, all_layer_outputs)
            - embeddings: Final node feature tensor of shape (N, FL).
            - all_layer_outputs: List of output tensors from every layer in order.
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]
        
    all_layer_outputs = []
    current_features = node_features
    
    # If activations is None, default to None for all layers
    if activations is None:
        activations = [None] * len(param_list)
        
    for idx, params in enumerate(param_list):
        weight = params['weight']
        bias = params.get('bias', None)
        activation = activations[idx] if idx < len(activations) else None
        
        # Run one full GCN layer forward pass
        current_features = gcn_layer_forward(
            node_features=current_features,
            src=src,
            dst=dst,
            weight=weight,
            bias=bias,
            num_nodes=num_nodes,
            activation=activation
        )
        
        all_layer_outputs.append(current_features)
        
    return current_features, all_layer_outputs

# Step 19 - gat_attention_logits
import torch
import torch.nn.functional as F

def gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight):
    """
    Scores every edge with unnormalized GAT attention.
    
    Args:
        node_features (Tensor): Node feature matrix of shape (N, Fin).
        src (LongTensor): Source node indices of shape (E,).
        dst (LongTensor): Destination node indices of shape (E,).
        attn_src (Tensor): Source attention vector of shape (Fout,).
        attn_dst (Tensor): Destination attention vector of shape (Fout,).
        weight (Tensor): Linear weight matrix of shape (Fin, Fout).
        
    Returns:
        tuple: (logits, transformed_nodes)
            - logits: Per-edge attention logits of shape (E,).
            - transformed_nodes: Transformed node matrix of shape (N, Fout).
    """
    # 1. Linear projection of node features: (N, Fin) @ (Fin, Fout) -> (N, Fout)
    transformed_nodes = torch.matmul(node_features, weight)
    
    # 2. Gather source and destination node features for all edges (E, Fout)
    h_src = gather_source_node_features(transformed_nodes, src)
    h_dst = gather_source_node_features(transformed_nodes, dst)
    
    # 3. Compute attention scores using dot products with attention vectors
    # attn_src^T h_src -> shape (E,)
    score_src = (h_src * attn_src).sum(dim=-1)
    # attn_dst^T h_dst -> shape (E,)
    score_dst = (h_dst * attn_dst).sum(dim=-1)
    
    # 4. Combine scores and apply LeakyReLU with negative_slope = 0.2
    logits = F.leaky_relu(score_src + score_dst, negative_slope=0.2)
    
    return logits, transformed_nodes

# Step 20 - gat_masked_neighbor_softmax
import torch

def gat_masked_neighbor_softmax(logits, dst, num_nodes):
    """
    Turns raw per-edge attention logits into attention coefficients that form 
    a valid softmax over each destination node's incoming neighbors only.
    
    Args:
        logits (Tensor): Per-edge attention logits of shape (E,).
        dst (LongTensor): Destination node indices of shape (E,).
        num_nodes (int): Total number of nodes N.
        
    Returns:
        Tensor: Attention coefficients of shape (E,) summing to 1 per destination node.
    """
    # 1. Numerical stability: Find max logit for each destination node
    max_logits = torch.full((num_nodes,), -float('inf'), device=logits.device, dtype=logits.dtype)
    max_logits.scatter_reduce_(0, dst, logits, reduce='amax', include_self=False)
    
    # Gather max logit back to each edge
    max_per_edge = max_logits[dst]
    
    # Subtract max for stability and compute exponential
    exp_logits = torch.exp(logits - max_per_edge)
    
    # 2. Compute sum of exponentials per destination node
    sum_logits = torch.zeros((num_nodes,), device=logits.device, dtype=logits.dtype)
    sum_logits.scatter_add_(0, dst, exp_logits)
    
    # Gather sum back to each edge
    sum_per_edge = sum_logits[dst]
    
    # 3. Normalize (add small epsilon to avoid division by zero)
    eps = 1e-16
    coefficients = exp_logits / (sum_per_edge + eps)
    
    return coefficients

# Step 21 - gat_head_forward
import torch

def gat_head_forward(node_features, src, dst, weight, attn_src, attn_dst, bias=None, activation=None, num_nodes=None):
    """
    Forward pass of a single GAT attention head.
    
    Args:
        node_features (Tensor): Node feature matrix of shape (N, Fin).
        src (LongTensor): Source node indices of shape (E,).
        dst (LongTensor): Destination node indices of shape (E,).
        weight (Tensor): Shared linear weight matrix of shape (Fin, Fout).
        attn_src (Tensor): Source attention vector of shape (Fout,).
        attn_dst (Tensor): Destination attention vector of shape (Fout,).
        bias (Tensor, optional): Bias vector of shape (Fout,).
        activation (callable, optional): Activation function applied after bias.
        num_nodes (int, optional): Total number of nodes N. Defaults to node_features.shape[0].
        
    Returns:
        tuple: (output_features, attention_coefficients)
            - output_features: Head output features of shape (N, Fout).
            - attention_coefficients: Per-edge attention coefficients of shape (E,).
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]
        
    # 1. Compute attention logits and transformed node features
    logits, transformed_nodes = gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight)
    
    # 2. Compute masked neighbor softmax to get attention coefficients
    coefficients = gat_masked_neighbor_softmax(logits, dst, num_nodes)
    
    # 3. Define message function: weight source node features by attention coefficients
    def gat_message_fn(x_src, x_dst, coeffs):
        return x_src * coeffs.unsqueeze(-1)
        
    # 4. Define update function: add optional bias
    def gat_update_fn(node_feats, aggregated):
        out = aggregated
        if bias is not None:
            out = out + bias
        if activation is not None:
            out = activation(out)
        return out
        
    # 5. Run message passing layer
    output_features = message_passing_layer(
        transformed_nodes,
        src,
        dst,
        message_fn=gat_message_fn,
        update_fn=gat_update_fn,
        aggr='sum',
        edge_attr=coefficients
    )
    
    return output_features, coefficients

# Step 22 - merge_gat_heads
import torch

def merge_gat_heads(heads, mode='concat'):
    """
    Merges multi-head GAT outputs into one node-feature tensor.
    
    Args:
        heads (list/tuple or Tensor): List of tensors [N, F] OR a stacked tensor [H, N, F].
        mode (str): 'concat' for concatenation [N, H*F] or 'mean' for averaging [N, F].
        
    Returns:
        Tensor: Merged node features of shape [N, H*F] or [N, F].
    """
    if mode not in ['concat', 'mean']:
        raise ValueError(f"Invalid mode '{mode}'. Must be 'concat' or 'mean'.")
        
    if isinstance(heads, (list, tuple)):
        if mode == 'concat':
            # Concatenate along the feature dimension
            return torch.cat(heads, dim=-1)
        elif mode == 'mean':
            # Stack into [H, N, F] then mean over the heads dimension (dim=0)
            return torch.stack(heads, dim=0).mean(dim=0)
            
    elif isinstance(heads, torch.Tensor):
        # heads is expected to be [H, N, F]
        if mode == 'concat':
            H, N, F = heads.shape
            # Transpose to [N, H, F], then flatten the last two dimensions to [N, H*F]
            return heads.transpose(0, 1).reshape(N, H * F)
        elif mode == 'mean':
            # Mean over the heads dimension (dim=0)
            return heads.mean(dim=0)
            
    else:
        raise TypeError("heads must be a list, tuple, or torch.Tensor.")

# Step 23 - gat_layer_forward
import torch

def gat_layer_forward(node_features, src, dst, param_list, merge_mode='concat', activation=None, num_nodes=None):
    """
    Executes one full multi-head Graph Attention (GAT) layer over a COO graph.
    
    Args:
        node_features (Tensor): Node feature matrix of shape (N, Fin).
        src (LongTensor): Source node indices of shape (E,).
        dst (LongTensor): Destination node indices of shape (E,).
        param_list (list of dict): Parameters for each head. Keys: 'weight', 'attn_src', 'attn_dst', ['bias'].
        merge_mode (str): 'concat' for intermediate layers, 'mean' for final layer.
        activation (callable, optional): Nonlinearity applied AFTER merging.
        num_nodes (int, optional): Total number of nodes N.
        
    Returns:
        tuple: (out, all_attn)
            - out: Merged node tensor of shape (N, F_merged).
            - all_attn: List of per-head attention coefficient tensors of shape (E,).
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]
        
    head_outputs = []
    all_attn = []
    
    # 1. Run each independent attention head
    for params in param_list:
        weight = params['weight']
        attn_src = params['attn_src']
        attn_dst = params['attn_dst']
        bias = params.get('bias', None)
        
        # Notice we do NOT pass the activation here; it happens after merging!
        out_feat, attn_coeffs = gat_head_forward(
            node_features, src, dst, weight, attn_src, attn_dst, 
            bias=bias, activation=None, num_nodes=num_nodes
        )
        
        head_outputs.append(out_feat)
        all_attn.append(attn_coeffs)
        
    # 2. Merge the head outputs
    merged_out = merge_gat_heads(head_outputs, mode=merge_mode)
    
    # 3. Apply the optional nonlinearity to the merged features
    if activation is not None:
        merged_out = activation(merged_out)
        
    return merged_out, all_attn

# Step 24 - init_gat_parameters
import torch
import math

def init_gat_parameters(in_dim, out_dim, num_heads, with_bias=True, seed=None):
    """
    Initializes parameters for a multi-head GAT layer using Glorot uniform initialization.
    
    Args:
        in_dim (int): Input feature dimension.
        out_dim (int): Output feature dimension per head.
        num_heads (int): Number of independent attention heads.
        with_bias (bool): Whether to include a bias vector.
        seed (int, optional): Random seed for reproducibility.
        
    Returns:
        list of dict: A list of length `num_heads`, each containing 'weight', 
                      'attn_src', 'attn_dst', and optionally 'bias' as float32 tensors with requires_grad=True.
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    # Calculate bounds for Glorot uniform initialization
    bound_w = math.sqrt(6.0 / (in_dim + out_dim))
    bound_a = math.sqrt(6.0 / (out_dim + 1))
    
    param_list = []
    
    for _ in range(num_heads):
        # 1. Initialize weight matrix
        weight = torch.empty(in_dim, out_dim, dtype=torch.float32)
        weight.uniform_(-bound_w, bound_w)
        weight.requires_grad = True
        
        # 2. Initialize source attention vector
        attn_src = torch.empty(out_dim, dtype=torch.float32)
        attn_src.uniform_(-bound_a, bound_a)
        attn_src.requires_grad = True
        
        # 3. Initialize destination attention vector
        attn_dst = torch.empty(out_dim, dtype=torch.float32)
        attn_dst.uniform_(-bound_a, bound_a)
        attn_dst.requires_grad = True
        
        # Pack into a dictionary
        params = {
            'weight': weight,
            'attn_src': attn_src,
            'attn_dst': attn_dst
        }
        
        # 4. Initialize optional bias
        if with_bias:
            bias = torch.zeros(out_dim, dtype=torch.float32)
            bias.requires_grad = True
            params['bias'] = bias
            
        param_list.append(params)
        
    return param_list

# Step 25 - gat_stack_forward
def gat_stack_forward(node_features, src, dst, layer_param_list, merge_modes=None, activations=None, num_nodes=None):
    if num_nodes is None:
        num_nodes = node_features.shape[0]
        
    num_layers = len(layer_param_list)
    
    # Set default merge_modes if none are provided
    if merge_modes is None:
        merge_modes = ['concat'] * (num_layers - 1) + ['mean']
        
    # Set default activations if none are provided
    if activations is None:
        activations = [None] * num_layers
        
    current_features = node_features
    all_layers_out = []
    
    for i in range(num_layers):
        # Run the multi-head GAT layer
        out = gat_layer_forward(
            current_features,
            src,
            dst,
            layer_param_list[i],
            merge_mode=merge_modes[i],
            activation=activations[i],
            num_nodes=num_nodes
        )
        
        # In case gat_layer_forward returns a tuple of (features, attention_weights)
        if isinstance(out, tuple):
            current_features = out[0]
        else:
            current_features = out
            
        # The prompt asks to save the intermediate node embeddings, not attention weights
        all_layers_out.append(current_features)
        
    return current_features, all_layers_out

# Step 26 - global_mean_pool
import torch

def global_mean_pool(node_features, batch_index, num_graphs=None):
    """
    Mean-pools node features into one graph-level vector per graph in a batch.
    
    Args:
        node_features (Tensor): Node feature matrix of shape (N, F).
        batch_index (LongTensor): Tensor of shape (N,) mapping nodes to graph IDs.
        num_graphs (int, optional): Total number of graphs (B).
        
    Returns:
        Tensor: Graph-level features of shape (B, F).
    """
    # 1. Infer the number of graphs if not provided
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1
        
    # 2. Sum the node features for each graph
    # We reuse scatter_sum_to_nodes, treating batch_index as the "dst" nodes
    summed_features = scatter_sum_to_nodes(node_features, batch_index, num_graphs)
    
    # 3. Count the number of nodes in each graph
    # We create a column of 1s and sum them up per graph
    ones = torch.ones_like(node_features[:, :1])  # Shape (N, 1)
    node_counts = scatter_sum_to_nodes(ones, batch_index, num_graphs)
    
    # 4. Divide sum by count to get the mean (clamp to prevent division by zero)
    mean_features = summed_features / node_counts.clamp(min=1)
    
    return mean_features

# Step 27 - global_sum_pool
import torch

def global_sum_pool(node_features, batch_index, num_graphs=None):
    """
    Sum-pools node features into one graph-level vector per graph in a batch.
    
    Args:
        node_features (Tensor): Node feature matrix of shape (N, F).
        batch_index (LongTensor): Tensor of shape (N,) mapping nodes to graph IDs.
        num_graphs (int, optional): Total number of graphs (B).
        
    Returns:
        Tensor: Graph-level features of shape (B, F).
    """
    # 1. Infer the number of graphs if not provided
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1
        
    # 2. Sum the node features for each graph
    # We reuse scatter_sum_to_nodes, treating batch_index as the "dst" nodes
    summed_features = scatter_sum_to_nodes(node_features, batch_index, num_graphs)
    
    return summed_features

# Step 28 - global_max_pool
import torch

def global_max_pool(node_features, batch_index, num_graphs=None):
    """
    Reduces node features into one max-pooled vector per graph in a batch.
    
    Args:
        node_features (Tensor): Node feature matrix of shape (N, F).
        batch_index (LongTensor): Tensor of shape (N,) mapping nodes to graph IDs.
        num_graphs (int, optional): Total number of graphs (B).
        
    Returns:
        Tensor: Graph-level features of shape (B, F).
    """
    # 1. Infer the number of graphs if not provided
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1
        
    # 2. Extract the maximum node feature for each graph
    # We reuse scatter_max_to_nodes, treating batch_index as the destination mapping
    max_features = scatter_max_to_nodes(node_features, batch_index, num_graphs)
    
    return max_features

# Step 29 - global_mean_max_pool
import torch

def global_mean_max_pool(node_features, batch_index, num_graphs=None):
    """
    Produces a richer graph-level embedding by concatenating mean and max pools.
    
    Args:
        node_features (Tensor): Node feature matrix of shape (N, F).
        batch_index (LongTensor): Tensor of shape (N,) mapping nodes to graph IDs.
        num_graphs (int, optional): Total number of graphs (B).
        
    Returns:
        Tensor: Concatenated graph-level features of shape (B, 2F).
    """
    # 1. Compute the mean pool (Shape: [B, F])
    mean_pooled = global_mean_pool(node_features, batch_index, num_graphs)
    
    # 2. Compute the max pool (Shape: [B, F])
    max_pooled = global_max_pool(node_features, batch_index, num_graphs)
    
    # 3. Concatenate along the feature dimension (Shape: [B, 2F])
    combined_features = torch.cat([mean_pooled, max_pooled], dim=-1)
    
    return combined_features

# Step 30 - node_classification_head
import torch

def node_classification_head(node_embeddings, weight, bias=None):
    """
    Maps per-node embeddings to class logits with a linear layer.
    
    Args:
        node_embeddings (Tensor): Final node embeddings of shape (N, H).
        weight (Tensor): Classification weight matrix of shape (H, C).
        bias (Tensor, optional): Bias vector of shape (C).
        
    Returns:
        Tensor: Class logits of shape (N, C).
    """
    # 1. Compute the matrix product: (N, H) @ (H, C) -> (N, C)
    logits = torch.matmul(node_embeddings, weight)
    
    # 2. Optionally add the bias
    if bias is not None:
        logits = logits + bias
        
    return logits

# Step 31 - graph_regression_head
import torch

def graph_regression_head(graph_embeddings, weight, bias=None):
    """
    Maps pooled graph-level embeddings to regression outputs.
    
    Args:
        graph_embeddings (Tensor): Pooled embeddings of shape [B, D].
        weight (Tensor): Regression weights of shape [out_dim, D].
        bias (Tensor, optional): Bias vector of shape [out_dim].
        
    Returns:
        Tensor: Predictions of shape [B, out_dim].
    """
    # 1. Transpose weight to shape [D, out_dim]
    weight_t = weight.t()
    
    # 2. Compute matrix product: [B, D] @ [D, out_dim] -> [B, out_dim]
    predictions = torch.matmul(graph_embeddings, weight_t)
    
    # 3. Optionally add bias
    if bias is not None:
        predictions = predictions + bias
        
    return predictions

# Step 32 - generate_sbm_graph
import torch

def generate_sbm_graph(num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        
    # Generate node features
    node_features = torch.randn(num_nodes, feature_dim)
    
    # Assign node labels in contiguous blocks
    node_labels = torch.zeros(num_nodes, dtype=torch.long)
    for c in range(num_classes):
        start = c * num_nodes // num_classes
        end = (c + 1) * num_nodes // num_classes
        node_labels[start:end] = c
        
    # Generate edges
    src = []
    dst = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prob = p_in if node_labels[i] == node_labels[j] else p_out
            if torch.rand(1).item() < prob:
                # Add both directions of the undirected edge immediately
                src.extend([i, j])
                dst.extend([j, i])
                
    # Create edge index tensor (will automatically be [2, 0] if lists are empty)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'node_labels': node_labels,
        'num_nodes': num_nodes
    }

# Step 33 - build_node_classification_dataset
import torch

def build_node_classification_dataset(num_graphs, num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    """
    Constructs a list of synthetic SBM graphs ready for node-classification training.
    
    Args:
        num_graphs (int): Number of graphs to generate for the dataset.
        num_nodes (int): Total number of nodes per graph.
        num_classes (int): Number of communities (classes).
        p_in (float): Probability of an edge inside a community.
        p_out (float): Probability of an edge across communities.
        feature_dim (int): Dimensionality of the node features.
        seed (int, optional): Base random seed for reproducibility.
        
    Returns:
        list of dict: A list of graph dictionaries in generation order.
    """
    dataset = []
    
    for i in range(num_graphs):
        # Derive a distinct per-graph seed so graphs differ, 
        # but the overall dataset remains reproducible!
        graph_seed = seed + i if seed is not None else None
        
        # Generate one SBM graph
        graph = generate_sbm_graph(
            num_nodes=num_nodes,
            num_classes=num_classes,
            p_in=p_in,
            p_out=p_out,
            feature_dim=feature_dim,
            seed=graph_seed
        )
        
        dataset.append(graph)
        
    return dataset

# Step 34 - generate_molecule_like_graph (not yet solved)
# TODO: implement

# Step 35 - build_graph_regression_dataset (not yet solved)
# TODO: implement

# Step 36 - collate_graph_batch (not yet solved)
# TODO: implement

# Step 37 - cross_entropy_loss (not yet solved)
# TODO: implement

# Step 38 - mse_loss (not yet solved)
# TODO: implement

# Step 39 - accuracy_metric (not yet solved)
# TODO: implement

# Step 40 - mae_metric (not yet solved)
# TODO: implement

# Step 41 - gnn_train_step (not yet solved)
# TODO: implement

# Step 42 - train_node_classifier (not yet solved)
# TODO: implement

# Step 43 - train_graph_regressor (not yet solved)
# TODO: implement

# Step 44 - representation_similarity (not yet solved)
# TODO: implement

# Step 45 - oversmoothing_diagnostic (not yet solved)
# TODO: implement

# Step 46 - mpnn_gnn_experiment (not yet solved)
# TODO: implement

