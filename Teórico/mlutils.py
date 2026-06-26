import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    #ax = plt.axes()
    #ax.set_aspect("equal", adjustable="datalim")
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.2)#OrRd_r)
    #plt.contourf(xx, yy, Z, colors = ["red","royalblue"], alpha=0.2)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=plt.cm.Spectral)
    
def plot_decision_boundary2(model, X, y):
    X = X.T
    y = y.T

    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax = plt.gca()
    ax.set_aspect(1)
    # plt.axis("equal")
    #plt.contourf(xx, yy, Z, cmap=plt.cm.ocean, alpha=0.5)
    plt.contourf(xx, yy, Z, colors = ["red","royalblue"], alpha=0.2)
    # plt.contourf(xx, yy, Z, cmap=plt.cm.Pastel1, alpha=0.5)
    # plt.scatter(X[0, y==1], X[1, y==1], color="dodgerblue", edgecolors='k', label="1")
    plt.scatter(X[0, y == 1], X[1, y == 1], color="royalblue", label="1")
    plt.scatter(X[0, y == -1], X[1, y == -1], color="red", label="-1")
    plt.legend()

def load_dataset():
    train_dataset = h5py.File('demo_6_dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('demo_6_dataset/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes



def load_dataset_disks(size, seed=39):
    """
    Genera un dataset con dos anillos concéntricos.
    
    Parámetros:
    -----------
    size : int
        Número total de muestras
    seed : int
        Semilla para reproducibilidad
    """
    def get_sample_disk(from_, to, size):
        length = np.random.uniform(from_, to, size)
        angle = np.pi * np.random.uniform(0, 2, size)
        x = length * np.cos(angle)
        y = length * np.sin(angle)
        return np.array(list(zip(x, y)))

    np.random.seed(seed)
    size1 = int(size / 2)
    size2 = size - size1
    X = np.concatenate((get_sample_disk(0, 1, size1), get_sample_disk(1, 2, size2)))
    Y = np.concatenate((np.ones(size1), -np.ones(size2)))
    return X, Y


def plot_kernel_comparison(X, Y, kernels, title=None, figsize=(20, 5)):
    """
    Visualiza la comparación de diferentes kernels.
    """
    n_kernels = len(kernels)
    fig, axes = plt.subplots(1, n_kernels, figsize=figsize)
    if n_kernels == 1:
        axes = [axes]
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold')
    
    for idx, (kernel_name, kernel_params) in enumerate(kernels.items()):
        ax = axes[idx]
        
        # Crear y entrenar clasificador
        if kernel_name == 'linear':
            clf = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0, random_state=42))
        else:
            clf = SVC(kernel=kernel_name, **kernel_params, random_state=42)
        
        clf.fit(X, Y)
        
        # Visualizar frontera
        DecisionBoundaryDisplay.from_estimator(
            clf, X, ax=ax, grid_resolution=200,
            plot_method='contour', colors='k',
            levels=[0], alpha=0.5,
            linestyles=['-']
        )
        
        # Graficar puntos
        ax.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=plt.cm.Spectral,
                   edgecolors='k', linewidth=0.5, alpha=0.8)
        
        # Mostrar vectores de soporte si están disponibles
        if hasattr(clf, 'support_vectors_'):
            ax.scatter(
                clf.support_vectors_[:, 0],
                clf.support_vectors_[:, 1],
                s=100, linewidth=2, facecolors='none', edgecolors='red',
                label='Vectores de Soporte'
            )
        
        # Métricas
        accuracy = accuracy_score(Y, clf.predict(X))
        n_support = len(clf.support_) if hasattr(clf, 'support_') else 0
        
        ax.set_title(f'{kernel_name}\nPrecisión: {accuracy:.3f}', fontweight='bold')
        ax.set_xlabel('$x_0$')
        ax.set_ylabel('$x_1$')
        ax.set_aspect('equal', adjustable='datalim')
        
        # Mostrar parámetros
        params_text = '\n'.join([f'{k}={v}' for k, v in kernel_params.items()])
        ax.text(0.02, 0.02, params_text, transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                verticalalignment='bottom')
        ax.text(0.02, 0.08, f'SV: {n_support}', transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7),
                verticalalignment='bottom')
    
    plt.tight_layout()
    return fig, axes
# FUNCIONES AUXILIARES

def plot_svr_results(X, y, y_pred, svr_model, title=None, ax=None):
    """Visualiza los resultados de SVR."""
    if ax is None:
        ax = plt.gca()
    
    # Asegurar que X tenga la forma correcta
    if X.ndim == 1:
        X_flat = X
    else:
        X_flat = X.flatten()
    
    # Ordenar para visualización suave
    idx = np.argsort(X_flat)
    X_sorted = X_flat[idx]
    y_sorted = y[idx]
    y_pred_sorted = y_pred[idx]
    
    # Datos reales
    ax.scatter(X_sorted, y_sorted, alpha=0.6, label='Datos reales', s=30)
    
    # Predicciones
    ax.plot(X_sorted, y_pred_sorted, 'b-', linewidth=2, label='Predicción SVR')
    
    # Margen epsilon
    if hasattr(svr_model, 'epsilon'):
        epsilon = svr_model.epsilon
        ax.fill_between(X_sorted, 
                        y_pred_sorted - epsilon, 
                        y_pred_sorted + epsilon, 
                        alpha=0.2, color='green', label=f'ε-tube (ε={epsilon})')
    
    # Vectores de soporte - corregido
    if hasattr(svr_model, 'support_vectors_'):
        if hasattr(svr_model, 'support_'):
            # Los índices de soporte están basados en el conjunto de entrenamiento
            # Solo podemos visualizarlos si tenemos acceso a los datos originales
            # En este caso, simplemente mostramos los vectores de soporte
            # como puntos en la gráfica
            if len(svr_model.support_vectors_) > 0:
                # Los vectores de soporte están en el espacio transformado
                # Para visualización, los mostramos como puntos destacados
                # en los datos de entrenamiento (aquí usamos los datos de prueba)
                ax.scatter(X_sorted, y_sorted, 
                          s=100, facecolors='none', edgecolors='green', 
                          linewidth=1, label='Vectores de soporte',
                          alpha=0.5)
    
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    # Métricas
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    metrics_text = f'R² = {r2:.3f}\nMSE = {mse:.3f}\nMAE = {mae:.3f}'
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if title:
        ax.set_title(title, fontweight='bold', fontsize=14)
    
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    return ax


def generate_synthetic_data(n_samples=200, noise=0.1, seed=42):
    """Genera datos sintéticos no lineales para regresión."""
    np.random.seed(seed)
    X = np.linspace(-5, 5, n_samples).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.3 * X.ravel() + noise * np.random.randn(n_samples)
    return X, y