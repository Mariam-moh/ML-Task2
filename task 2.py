from sklearn.decomposition import PCA
import numpy as np


shipping_data = np.array([
    [10, 300, 24],   
    [15, 500, 36],   
    [8, 250, 20],   
    [20, 700, 48]    
])

pca = PCA(n_components=3)

shipping_pca = pca.fit_transform(shipping_data)

print("Original Shipping Data:")
print("Weight(kg)   Distance(km)   Time(hours)")
print(shipping_data)

print("\nData after PCA:")
print(shipping_pca)

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print(f"\nCumulative variance: {np.cumsum(pca.explained_variance_ratio_)}")
