def goodfellow_backprop_ggn(activations, linearGrads):
    """
    Returns the mean of the gradients and the diagonal of the GGN as a list of tensors
    """
    grads = []
    diagGGNs = []
    for i in range(len(linearGrads)):
        G, X = linearGrads[i], activations[i]
        if len(G.shape) < 2:
            G = G.unsqueeze(1)

        G *= G.shape[0] # if the function is an average

        #pdb.set_trace()
        grads.append(G.t() @ X)
        grads.append(G.sum(dim=0))
        diagGGNs.append(G.t()**2 @ X**2)
        diagGGNs.append((G**2).sum(dim=0))
    return grads, diagGGNs
