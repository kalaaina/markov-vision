import numpy as np

def init_labels(image, n_classes):
    """
    Initialisation des classes par K-means maison (pur NumPy).
    
    Paramètres:
        image : array (H, W) ou (H, W, 3) - image d'entrée
        n_classes : int - nombre de segments souhaités
    Retour:
        labels : array (H, W) d'entiers 0..n_classes-1
    """
    h, w = image.shape[:2]
    
    # Mettre les pixels sous forme 2D (N_pixels, features)
    if len(image.shape) == 3:
        pixels = image.reshape(-1, 3).astype(np.float32)
    else:
        pixels = image.reshape(-1, 1).astype(np.float32)
    
    # Initialisation aléatoire des centres
    np.random.seed(42)  # pour reproductibilité
    indices = np.random.choice(len(pixels), n_classes, replace=False)
    centers = pixels[indices]
    
    # K-means : max 10 itérations
    for _ in range(10):
        # Distance aux centres
        distances = np.zeros((len(pixels), n_classes))
        for k in range(n_classes):
            distances[:, k] = np.linalg.norm(pixels - centers[k], axis=1)
        
        # Assignation
        labels_flat = np.argmin(distances, axis=1)
        
        # Recalcul des centres
        new_centers = []
        for k in range(n_classes):
            if np.sum(labels_flat == k) > 0:
                new_centers.append(pixels[labels_flat == k].mean(axis=0))
            else:
                new_centers.append(centers[k])
        new_centers = np.array(new_centers)
        
        # Convergence ?
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    
    # Remise en forme 2D
    labels = labels_flat.reshape(h, w)
    return labels


if __name__ == '__main__':
    # Test conforme au guide (page 2)
    fake_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    labels = init_labels(fake_image, n_classes=3)
    
    print('✅ TOBOSSI - initializer.py OK')
    print('Shape labels:', labels.shape)
    print('Classes presentes:', np.unique(labels))
    
    import matplotlib.pyplot as plt
    plt.imshow(labels, cmap='tab10')
    plt.title('Labels initiaux (K-means)')
    plt.colorbar()
    plt.show()