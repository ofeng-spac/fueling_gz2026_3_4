# flb2_utils/validation.py
import numpy as np

def validate_intersection_methods(idx1, pts0_1, pts1_1, idx2, pts0_2, pts1_2):
    """验证两种交集方法的结果是否一致"""
    print("\n=== 方法验证 ===")
    
    count_same = len(idx1) == len(idx2)
    idx_match = np.array_equal(np.sort(idx1), np.sort(idx2)) if count_same else False
    pts_match = np.allclose(pts0_1, pts0_2) and np.allclose(pts1_1, pts1_2) if count_same else False
    
    if count_same and idx_match and pts_match:
        print("两种方法结果完全一致")
    else:
        print("两种方法结果不一致")
        if not count_same:
            print(f"  数量差异: 方法1={len(idx1)}, 方法2={len(idx2)}")
        elif not idx_match:
            print(f"  索引差异")
        elif not pts_match:
            print("  点坐标有微小差异")
    
    return {
        "count_same": count_same,
        "idx_match": idx_match,
        "pts_match": pts_match,
        "count_different": not (count_same and idx_match and pts_match)
    }

def verify_essential_matrix(E):
    """验证本质矩阵的性质"""
    print("\n=== 本质矩阵验证 ===")
    print("本质矩阵 E:")
    print(E)
    
    rank = np.linalg.matrix_rank(E)
    print(f"秩 (应该是2): {rank}")
    
    U, S, Vt = np.linalg.svd(E)
    print(f"奇异值: {S}")
    
    if rank == 2:
        ratio = S[0] / S[1] if S[1] != 0 else float('inf')
        print(f"奇异值比值 S[0]/S[1]: {ratio:.6f}")
    
    det = np.linalg.det(E)
    print(f"行列式 (应该接近0): {det:.10f}")
    
    return rank, S, det
