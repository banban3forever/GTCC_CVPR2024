import torch
import numpy as np
from numba import jit

def align_two_videos(model, video_A, video_B, num_output_frames=None):
    """
    对齐两个不同长度的视频
    
    Args:
        model: 训练好的 GTCC 模型
        video_A: shape (M, feature_dim) 或 (M, C, H, W)
        video_B: shape (N, feature_dim) 或 (N, C, H, W)
        num_output_frames: 输出帧数 X，默认为 min(M, N)
    
    Returns:
        aligned_indices_A: A' 的帧索引 (长度 X)
        aligned_indices_B: B' 的帧索引 (长度 X)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        # 1. 获取嵌入特征
        embed_A = model([video_A.to(device)])['outputs'][0]  # (M, embed_dim)
        embed_B = model([video_B.to(device)])['outputs'][0]  # (N, embed_dim)
        
        M, N = embed_A.shape[0], embed_B.shape[0]
        
        # 2. 计算帧间距离矩阵
        dist_matrix = torch.cdist(embed_A, embed_B)  # (M, N)
        
        # 3. 找到最近邻匹配
        # A->B: 对于 A 中每一帧，找 B 中最近的帧
        nn_A_to_B = dist_matrix.argmin(dim=1).cpu().numpy()  # (M,)
        # B->A: 对于 B 中每一帧，找 A 中最近的帧
        nn_B_to_A = dist_matrix.argmin(dim=0).cpu().numpy()  # (N,)
        
        # 4. 获取双向匹配（互为最近邻的配对，更可靠）
        mutual_matches = []
        for i in range(M):
            j = nn_A_to_B[i]
            if nn_B_to_A[j] == i:  # 互为最近邻
                mutual_matches.append((i, j))
        
        # 5. 如果没有足够的互匹配，使用单向匹配
        if len(mutual_matches) < 10:
            # 使用 A->B 的匹配
            matches = [(i, nn_A_to_B[i]) for i in range(M)]
        else:
            matches = mutual_matches
        
        # 6. 选择 X 个均匀分布的配对
        if num_output_frames is None:
            num_output_frames = min(M, N)
        
        # 均匀采样
        indices = np.linspace(0, len(matches) - 1, num_output_frames, dtype=int)
        selected_matches = [matches[i] for i in indices]
        
        aligned_indices_A = np.array([m[0] for m in selected_matches])
        aligned_indices_B = np.array([m[1] for m in selected_matches])
        
    return aligned_indices_A, aligned_indices_B


def get_aligned_videos(video_A, video_B, aligned_indices_A, aligned_indices_B):
    """
    根据对齐索引提取配对帧
    
    Returns:
        A_prime: 对齐后的视频 A (X 帧)
        B_prime: 对齐后的视频 B (X 帧)
    """
    A_prime = video_A[aligned_indices_A]
    B_prime = video_B[aligned_indices_B]
    return A_prime, B_prime


# DTW
def dtw_alignment(embed_A, embed_B):
    """
    使用 DTW 获取单调对齐路径
    
    Args:
        embed_A: (M, D) 视频A的嵌入
        embed_B: (N, D) 视频B的嵌入
    
    Returns:
        path: list of (i, j) 配对索引
    """
    # 计算距离矩阵
    dist_matrix = torch.cdist(embed_A, embed_B).cpu().numpy()  # (M, N)
    
    M, N = dist_matrix.shape
    
    # DTW 累积代价矩阵
    D = np.full((M + 1, N + 1), np.inf)
    D[0, 0] = 0
    
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            cost = dist_matrix[i-1, j-1]
            D[i, j] = cost + min(D[i-1, j],      # 垂直 (insertion)
                                  D[i, j-1],      # 水平 (deletion)  
                                  D[i-1, j-1])    # 对角 (match)
    
    # 回溯获取对齐路径
    path = []
    i, j = M, N
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        candidates = [
            (D[i-1, j-1], i-1, j-1),  # 对角
            (D[i-1, j], i-1, j),       # 垂直
            (D[i, j-1], i, j-1)        # 水平
        ]
        _, i, j = min(candidates, key=lambda x: x[0])
    
    path.reverse()
    return path


def align_videos_with_dtw(model, video_A, video_B, num_output_frames=None):
    """
    使用 DTW 对齐两个视频，保证单调性（解决动作重复问题）
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    with torch.no_grad():
        embed_A = model([video_A.to(device)])['outputs'][0].cpu()
        embed_B = model([video_B.to(device)])['outputs'][0].cpu()
    
    # DTW 对齐
    dtw_path = dtw_alignment(embed_A, embed_B)
    
    # DTW 路径可能很长（因为有重复），需要采样
    if num_output_frames is None:
        num_output_frames = min(embed_A.shape[0], embed_B.shape[0])
    
    # 均匀采样 X 个配对
    indices = np.linspace(0, len(dtw_path) - 1, num_output_frames, dtype=int)
    sampled_path = [dtw_path[i] for i in indices]
    
    aligned_indices_A = np.array([p[0] for p in sampled_path])
    aligned_indices_B = np.array([p[1] for p in sampled_path])
    
    return aligned_indices_A, aligned_indices_B