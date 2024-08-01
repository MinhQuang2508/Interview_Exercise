#a) Viết công thức toán, implement (code numpy) và giải thích về Triplet loss.
import numpy as np

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))
 
def triplet_loss(anchor, positive, negative, alpha=0.2):
    """
    Compute the triplet loss.
    
    Args:
    anchor -- numpy array of shape (d,) for anchor point
    positive -- numpy array of shape (d,) for positive point
    negative -- numpy array of shape (d,) for negative point
    alpha -- margin value
    
    Returns:
    loss -- triplet loss
    """
    d_ap = euclidean_distance(anchor, positive)
    d_an = euclidean_distance(anchor, negative)
   
    loss = np.maximum(d_ap - d_an + alpha, 0)
    return loss
#b). Viêt công thức toán, implement (code numpy) và giải thích khi Input triplet loss mở rộng không chỉ là 1 mẫu thật và một mẫu giả nữa mà sẽ là 2 mẫu thật và 5 mẫu giả.
 
def multi_class_triplet_loss(anchor, positive1, positive2, negatives, alpha=0.2):
    """
    Compute the triplet loss with multi class triplet loss.
    
    Args:
    anchor -- numpy array of shape (d,) for anchor point
    positive -- numpy array of shape (d,) for positive point
    negatives -- list of numpy arrays, each of shape (d,) for negative points
    alpha -- margin value
    
    Returns:
    loss -- triplet loss
    """
    d_ap1 = euclidean_distance(anchor, positive1)
    d_ap2 = euclidean_distance(anchor, positive2)
 
    d_an = np.array([euclidean_distance(anchor, n) for n in negatives])
    loss_pos1 = np.maximum(d_ap1 - d_an + alpha, 0)
    loss_pos2 = np.maximum(d_ap2 - d_an + alpha, 0)
 
    max_loss_pos1 = np.max(loss_pos1)
    max_loss_pos2 = np.max(loss_pos2)
   
    total_loss = max_loss_pos1 + max_loss_pos2
    return total_loss
 
#example
anchor = np.array([1.0, 2.0])
positive1 = np.array([1.1, 2.1])
positive2 = np.array([1.2, 2.2])
negatives = [np.array([3.0, 4.0]), np.array([3.1, 4.1]), np.array([3.2, 4.2]),
             np.array([3.3, 4.3]), np.array([3.4, 4.4])]
loss = multi_class_triplet_loss(anchor, positive1, positive2, negatives)
loss1 = triplet_loss(anchor,positive1,negatives[0])
print("Multi-class Triplet Loss:", loss)
print(loss1)