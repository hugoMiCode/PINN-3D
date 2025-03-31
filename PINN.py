import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Si mis a True cela va charger PINN_model.keras
load_previous_model = False  # True si on veut charger un modèle préalablement entraîné
model_name = "PINN_model_2000.keras"  # Nom du modèle à charger

# -----------------------------
# Paramètres du problème
# -----------------------------
k = 0.1  # diffusivité thermique
lb = np.array([0.0, 0.0, 0.0])   # bornes inférieures pour (x, y, t)
ub = np.array([1.0, 1.0, 1.0])   # bornes supérieures pour (x, y, t)

# Nombre de points pour chaque ensemble
N_f = 8000    # points de collocation pour la PDE
N_0 = 800      # points pour la condition initiale (t = 0)
N_b = 400      # points sur le contour (conditions aux limites)
N_fixed = 60  # points pour la contrainte fixe au point (0,0,t)

# -----------------------------
# Génération des points d'entraînement
# -----------------------------

# Points de collocation dans le domaine (x, y, t)
X_f = np.random.rand(N_f, 3)

# --- Condition initiale ---
# Pour t = 0, on tire aléatoirement (x,y) dans [0,1] et on impose :
# u(x,y,0) = 15°C pour y>=0.8 (haut du carré) et 0°C sinon.
x0 = np.random.rand(N_0, 1)
y0 = np.random.rand(N_0, 1)
t0 = np.zeros((N_0, 1))
X0 = np.concatenate([x0, y0, t0], axis=1)
u0 = np.where(y0 >= 0.8, 1, 0.0)

# --- Conditions aux limites ---
# On impose u = 0 sur le contour du carré ([0,1] x [0,1])
# On génère des points sur les bords : x=0, x=1, y=0 et y=1, pour t aléatoire dans [0,1]
xb1 = np.zeros((N_b//4, 1))   # x = 0
xb2 = np.ones((N_b//4, 1))    # x = 1
yb = np.random.rand(N_b//4, 1)
tb = np.random.rand(N_b//4, 1)
X_b1 = np.concatenate([xb1, yb, tb], axis=1)
X_b2 = np.concatenate([xb2, yb, tb], axis=1)

yb1 = np.zeros((N_b//4, 1))   # y = 0
yb2 = np.ones((N_b//4, 1))    # y = 1
xb = np.random.rand(N_b//4, 1)
tb2 = np.random.rand(N_b//4, 1)
X_b3 = np.concatenate([xb, yb1, tb2], axis=1)
X_b4 = np.concatenate([xb, yb2, tb2], axis=1)
X_b = np.concatenate([X_b1, X_b2, X_b3, X_b4], axis=0)
u_b = np.zeros((X_b.shape[0], 1))

# --- Contrainte sur le point fixe ---
# Pour tout t, on impose que le point (0,0,t) ait u=10°C.
t_fixed = np.random.rand(N_fixed, 1)  # instants t aléatoires dans [0,1]
X_fixed = np.concatenate([np.zeros((N_fixed, 1)),
                          np.zeros((N_fixed, 1)),
                          t_fixed], axis=1)
u_fixed = 0.5 * np.ones((N_fixed, 1))

# Conversion en tenseurs TensorFlow
X_f_tf = tf.convert_to_tensor(X_f, dtype=tf.float32)
X0_tf  = tf.convert_to_tensor(X0, dtype=tf.float32)
u0_tf  = tf.convert_to_tensor(u0, dtype=tf.float32)
X_b_tf = tf.convert_to_tensor(X_b, dtype=tf.float32)
u_b_tf = tf.convert_to_tensor(u_b, dtype=tf.float32)
X_fixed_tf = tf.convert_to_tensor(X_fixed, dtype=tf.float32)
u_fixed_tf = tf.convert_to_tensor(u_fixed, dtype=tf.float32)

# -----------------------------
# Définition du modèle PINN
# -----------------------------
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(shape=(3,)),  # entrée (x, y, t)
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(50, activation='tanh'),
        tf.keras.layers.Dense(1, activation=None)   # sortie u(x,y,t)
    ])
    return model

if load_previous_model:
    print("Chargement du modèle préalablement entraîné...")
    model = tf.keras.models.load_model(model_name)
else:
    model = build_model()

# -----------------------------
# Définition du résidu de la PDE
# -----------------------------
def pde_residual(model, X):
    """
    Calcule le résidu de l'équation de la chaleur : u_t - k*(u_xx + u_yy)
    """
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(X)
        with tf.GradientTape(persistent=True) as tape1:
            tape1.watch(X)
            u = model(X)
        grads = tape1.gradient(u, X)
        u_x = grads[:, 0:1]
        u_y = grads[:, 1:2]
        u_t = grads[:, 2:3]
    u_xx = tape2.gradient(u_x, X)[:, 0:1]
    u_yy = tape2.gradient(u_y, X)[:, 1:2]
    del tape1, tape2
    return u_t - k*(u_xx + u_yy)

# -----------------------------
# Définition de la fonction de perte totale
# -----------------------------
def loss_total():
    # Perte sur la condition initiale
    u0_pred = model(X0_tf)
    loss_ic = tf.reduce_mean(tf.square(u0_tf - u0_pred))
    
    # Perte sur les conditions aux limites
    u_b_pred = model(X_b_tf)
    loss_bc = tf.reduce_mean(tf.square(u_b_tf - u_b_pred))
    
    # Perte sur le résidu de la PDE
    f_pred = pde_residual(model, X_f_tf)
    loss_pde = tf.reduce_mean(tf.square(f_pred))
    
    # Perte sur la contrainte du point fixe (0,0,t)
    u_fixed_pred = model(X_fixed_tf)
    loss_fixed = tf.reduce_mean(tf.square(u_fixed_tf - u_fixed_pred))
    
    total_loss = loss_ic + loss_bc + loss_pde + loss_fixed
    return total_loss, loss_ic, loss_bc, loss_pde, loss_fixed

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# -----------------------------
# Boucle d'entraînement
# -----------------------------
if not load_previous_model:
    epochs = 2000
    loss_history = []
    loss_ic_history = []
    loss_bc_history = []
    loss_pde_history = []
    loss_fixed_history = []

    for epoch in range(epochs + 1):
        with tf.GradientTape() as tape:
            loss_value, loss_ic, loss_bc, loss_pde, loss_fixed = loss_total()
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Total Loss: {loss_value.numpy():.5e}, "
                f"IC: {loss_ic.numpy():.5e}, BC: {loss_bc.numpy():.5e}, "
                f"PDE: {loss_pde.numpy():.5e}, Fixed: {loss_fixed.numpy():.5e}")


        # Stockage des valeurs de loss à chaque époque
        loss_history.append(loss_value.numpy())
        loss_ic_history.append(loss_ic.numpy())
        loss_bc_history.append(loss_bc.numpy())
        loss_pde_history.append(loss_pde.numpy())
        loss_fixed_history.append(loss_fixed.numpy())


    model.save("PINN_model.keras")  # Enregistrement du modèle


# -----------------------------
# Affichage statique pour 4 instants (exemple : t = 0, 0.33, 0.66, 1.0)
# -----------------------------
times_to_plot = [0.0, 0.33, 0.66, 1.0]
N_plot = 200
x = np.linspace(0, 1, N_plot)
y = np.linspace(0, 1, N_plot)
X_grid, Y_grid = np.meshgrid(x, y)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Boucle sur les sous-graphiques
for i in range(2):
    for j in range(2):
        idx = i * 2 + j
        if idx < len(times_to_plot):
            t_val = times_to_plot[idx]
            T_grid = t_val * np.ones_like(X_grid)
            X_plot = np.stack([X_grid.flatten(), Y_grid.flatten(), T_grid.flatten()], axis=1)
            u_pred = model(tf.convert_to_tensor(X_plot, dtype=tf.float32))
            U_pred = u_pred.numpy().reshape(N_plot, N_plot)
            ax = axes[i, j]
            cp = ax.contourf(X_grid, Y_grid, U_pred, 100, cmap='hot')
            fig.colorbar(cp, ax=ax)
            ax.set_title(f"t = {t_val:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

# Réglages supplémentaires pour la figure
fig.suptitle(
    "Équation de la chaleur : u_t = 0.1 (u_xx + u_yy)\n"
    "IC: u(x,y,0)=15°C pour y≥0.8, 0°C sinon | BC: u=0 sur le contour\n"
    "Contrainte fixe: u(0,0,t)=0.5°C pour tout t",
    fontsize=12
)
plt.tight_layout()
plt.savefig("solution_plot.png")  # Enregistrement de l'image
plt.show()



# -----------------------------
# Affichage de l'évolution de la loss
# -----------------------------
if not load_previous_model:
    log_loss = np.log(loss_history)  # calcul du logarithme de la loss
    plt.figure(figsize=(8, 5))
    plt.plot(log_loss, label="Log(Total Loss)")
    plt.xlabel("Époque")
    plt.ylabel("log(Loss)")
    plt.title("Évolution du Log de la Loss pendant l'entraînement")
    plt.legend()
    plt.grid(True)
    plt.savefig("log_loss_evolution.png")  # Enregistrement de l'image
    plt.show()

    # On affiche les pertes individuelles
    # (IC, BC, PDE, Fixed) sur 4 sous-graphes
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    
    losses = [loss_ic_history, loss_bc_history, loss_pde_history, loss_fixed_history]
    titles = ["Log(Loss IC)", "Log(Loss BC)", "Log(Loss PDE)", "Log(Loss Fixed)"]
    
    for ax, loss, title in zip(axes, losses, titles):
        ax.plot(loss)
        ax.set_title(title)
        ax.set_xlabel("Époque")
        ax.set_ylabel("log(Loss)")
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("losses_grid.png")
    plt.show()


# -----------------------------
# Animation de l'évolution en continu
# -----------------------------
fig2, ax2 = plt.subplots()
# Initialisation avec une image vide
im = ax2.imshow(np.zeros((N_plot, N_plot)), extent=[0, 1, 0, 1], origin='lower', cmap='hot')
ax2.set_xlabel("x")
ax2.set_ylabel("y")
cb = fig2.colorbar(im, ax=ax2)

def update(frame):
    t_val = frame
    T_grid = t_val * np.ones_like(X_grid)
    X_plot = np.stack([X_grid.flatten(), Y_grid.flatten(), T_grid.flatten()], axis=1)
    u_pred = model(tf.convert_to_tensor(X_plot, dtype=tf.float32))
    U_pred = u_pred.numpy().reshape(N_plot, N_plot)
    im.set_data(U_pred)
    ax2.set_title(f"t = {t_val:.2f}")
    return [im]


# On crée 120 frames pour t variant de 0 à maxTime
maxTime = 8
frames = np.linspace(0, maxTime, 120)
ani = animation.FuncAnimation(fig2, update, frames=frames, blit=True, interval=100)

ani.save(f"animation_t{maxTime}.gif", writer="pillow", fps=24)

plt.show()

# -----------------------------
# Calcul et affichage de la loss totale (sur la grille 2D) en fonction du temps
# -----------------------------
loss_total_norm = []
time_vals = frames  # mêmes instants que pour l'animation

for t_val in time_vals:
    # Construction de la grille pour un temps donné
    T_grid = t_val * np.ones_like(X_grid)
    X_plot = np.stack([X_grid.flatten(), Y_grid.flatten(), T_grid.flatten()], axis=1)
    
    # Prédiction du modèle sur la grille
    u_pred = model(tf.convert_to_tensor(X_plot, dtype=tf.float32))
    U_pred = u_pred.numpy().reshape(N_plot, N_plot)
    
    # --- Loss PDE sur la grille ---
    residual = pde_residual(model, tf.convert_to_tensor(X_plot, dtype=tf.float32))
    loss_pde = tf.reduce_mean(tf.square(residual)).numpy()
    
    # --- Loss sur la contrainte fixe au point (0,0,t) ---
    u_fixed_pred = model(tf.convert_to_tensor([[0.0, 0.0, t_val]], dtype=tf.float32))
    loss_fixed = tf.square(u_fixed_pred - 0.5).numpy()[0][0]
    
    # --- Loss pour les conditions aux limites ---
    # On extrait les bords de la grille :
    U_top = U_pred[0, :]      # y = 0
    U_bottom = U_pred[-1, :]   # y = 1
    U_left = U_pred[:, 0]      # x = 0
    U_right = U_pred[:, -1]    # x = 1
    loss_bc_top = np.mean(np.square(U_top))
    loss_bc_bottom = np.mean(np.square(U_bottom))
    loss_bc_left = np.mean(np.square(U_left))
    loss_bc_right = np.mean(np.square(U_right))
    loss_bc = (loss_bc_top + loss_bc_bottom + loss_bc_left + loss_bc_right) / 4.0
    
    # --- Loss pour la condition initiale ---
    # La condition initiale est définie pour t=0 : u(x,y,0)=1 si y>=0.8, 0 sinon.
    # Pour t très proche de 0 (on peut fixer une tolérance), on évalue l'erreur.
    if np.abs(t_val) < 1e-3:
        u0_exact = np.where(Y_grid >= 0.8, 1.0, 0.0)
        loss_ic = np.mean(np.square(u0_exact - U_pred))
    else:
        loss_ic = 0.0

    # --- Loss totale sur la grille ---
    total_loss = loss_pde + loss_fixed + loss_bc + loss_ic
    loss_total_norm.append(total_loss)

# Affichage du graphique de la loss totale en fonction du temps
plt.figure(figsize=(8, 5))
plt.plot(time_vals, loss_total_norm, label='Normalized Total Loss')
plt.xlabel("Temps t")
plt.ylabel("Loss totale normalisée")
plt.title("Évolution de la loss totale normalisée en fonction du temps")
plt.legend()
plt.grid(True)
plt.savefig("normalized_total_loss_vs_time.png")
plt.show()
