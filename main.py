import base64
import os
import struct
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk, ImageOps
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

IMAGE_DISPLAY_SIZE = (420, 420)
CONTROL_FONT_BOLD = ('Segoe UI', 10, 'bold')
CONTROL_FONT = ('Segoe UI', 10)
SLIDER_LENGTH = 220
BUTTON_PAD_X = 8
BUTTON_PAD_Y = 8


def _labels_to_rgb(labels, n_classes):
    """Convertit une carte d'étiquettes (H,W) en image RGB (uint8).
    Utilise une palette fixe adaptée jusqu'à 6 classes.
    """
    palette = [
        [255, 100, 100],
        [100, 200, 100],
        [100, 150, 255],
        [255, 220, 80],
        [200, 100, 200],
        [180, 180, 180],
    ]

    H, W = labels.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)

    for k in range(min(n_classes, len(palette))):
        rgb[labels == k] = palette[k]

    # Si plus de classes que la palette, assigner des couleurs aléatoires
    if n_classes > len(palette):
        for k in range(len(palette), n_classes):
            color = np.random.randint(0, 256, size=3)
            rgb[labels == k] = color

    return rgb


def _labels_to_natural_rgb(labels, patch):
    """Mappe les labels synthétisés aux couleurs naturelles moyennes du patch d'origine.
    Cela permet à la texture générée de conserver l'apparence naturelle (ex: bois) du patch.
    """
    n_classes = int(labels.max() + 1)
    H, W = labels.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    
    # 1. Convertir le patch en niveaux de gris pour reproduire la discrétisation de texture_synth.py
    if patch.ndim == 3:
        patch_gray = (0.299 * patch[:, :, 0] +
                      0.587 * patch[:, :, 1] +
                      0.114 * patch[:, :, 2]).astype(np.float32)
    else:
        patch_gray = patch.astype(np.float32)
        
    # 2. Obtenir les tranches de labels du patch d'origine
    bins = np.linspace(patch_gray.min(), patch_gray.max() + 1, n_classes + 1)
    patch_labels = np.digitize(patch_gray, bins[:-1]) - 1
    patch_labels = np.clip(patch_labels, 0, n_classes - 1)
    
    # 3. Calculer la couleur moyenne (R, G, B) de chaque classe dans le patch d'origine
    for k in range(n_classes):
        mask_patch = (patch_labels == k)
        if np.sum(mask_patch) > 0:
            mean_color = patch[mask_patch].mean(axis=0).astype(np.uint8)
        else:
            # Fallback si une classe n'a pas de pixels (très rare)
            palette = [
                [255, 100, 100], [100, 200, 100], [100, 150, 255],
                [255, 220, 80], [200, 100, 200], [180, 180, 180]
            ]
            mean_color = palette[k % len(palette)]
            
        # Appliquer cette couleur naturelle moyenne aux labels correspondants du canevas
        mask_labels = (labels == k)
        rgb[mask_labels] = mean_color
        
    return rgb


def log(msg: str):
    """Append a timestamped message to the UI log (thread-safe)."""
    ts = time.strftime('%H:%M:%S')

    def _append():
        try:
            log_widget.configure(state='normal')
            log_widget.insert('end', f'[{ts}] {msg}\n')
            log_widget.see('end')
            log_widget.configure(state='disabled')
        except Exception:
            pass

    if 'root' in globals():
        try:
            root.after(0, _append)
        except Exception:
            # fallback to printing
            print(f'[{ts}] {msg}')
    else:
        print(f'[{ts}] {msg}')


from config import BETA, TEMPERATURE, N_ITERATIONS, N_CLASSES, IMAGE_SIZE



# --- Nouveaux imports (Le travail de l'équipe) ---
import config
import image_loader
import initializer
import gibbs_sampler
import texture_synth
import convergence

# --- Variables d'état globales pour le live update ---
current_image = None       # Image originale NumPy
initial_labels = None      # Labels de départ (K-means)
current_labels = None      # Labels au cours du Gibbs
gibbs_thread = None        # Référence vers le thread Gibbs actif
abort_event = threading.Event()  # Événement d'annulation du thread
debounce_timer = None      # ID du timer de debouncing Tkinter
active_workflow = 'A'      # Canal actif : 'A' (segmentation) ou 'B' (texture)
current_patch = None       # Patch de texture original pour le Workflow B





ICON_PNG_B64 = (
    'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAQAAAC1+jfqAAAAHklEQVR4AWMYBaNgFIwMDAwGJgYGBgaGhgYGBgAAAwA'
    'A8v0G7n0AAAAAElFTkSuQmCC'
)

LANGUAGES = {
    'fr': {
        'beta': "Paramètre Beta (Cohésion):",
        'temperature': "Paramètre T (Température):",
        'workflow_a': "Lancer Workflow A",
        'workflow_b': "Lancer Workflow B",
        'status': "Redimensionnez la fenêtre pour agrandir l’interface proportionnellement.",
        'image': "[ L'image s'affichera ici ]",
        'switch_theme': "Basculer le thème",
        'switch_lang': "EN",
        'toggle_logs': "Afficher/Cacher les logs",
        'show_logs': "Afficher logs",
        'hide_logs': "Cacher logs",
        'plot_title': "Convergence (Énergie)",
    },
    'en': {
        'beta': "Beta (Cohesion):",
        'temperature': "T (Temperature):",
        'workflow_a': "Run Workflow A",
        'workflow_b': "Run Workflow B",
        'status': "Resize window to scale UI proportionally.",
        'image': "[ Image will appear here ]",
        'switch_theme': "Toggle Theme",
        'switch_lang': "FR",
        'toggle_logs': "Toggle logs",
        'show_logs': "Show logs",
        'hide_logs': "Hide logs",
        'plot_title': "Convergence (Energy)",
    },
}

THEMES = {
    'dark': {
        'root_bg': '#121821',
        'main_bg': '#121821',
        'frame_bg': '#1e2735',
        'fg': '#d7e2ef',
        'primary': '#3c82f6',
        'primary_hover': '#5c9cff',
        'secondary': '#2c3e50',
        'secondary_hover': '#3d5573',
        'trough': '#2a3649',
        'status_fg': '#c5d4e0',
    },
    'light': {
        'root_bg': '#f4f7fb',
        'main_bg': '#f4f7fb',
        'frame_bg': '#ffffff',
        'fg': '#1f2933',
        'primary': '#2b6ef6',
        'primary_hover': '#5c9cff',
        'secondary': '#3b5568',
        'secondary_hover': '#6a8396',
        'trough': '#d1d9e0',
        'status_fg': '#3b5568',
    },
}


def make_callback(root, label_widget):
    """Crée le callback thread-safe pour mettre à jour la segmentation et le tracé de convergence en direct."""
    def _cb(current_labels, iteration):
        if abort_event.is_set():
            return
            
        # Convertir les labels actuels en image RGB pour l'affichage
        rgb = _labels_to_rgb(current_labels, N_CLASSES)
        im = Image.fromarray(rgb)
        photo = build_photo_image(im)

        # Mise à jour de l'image de segmentation et du statut via root.after (sécurité thread)
        def _update():
            if abort_event.is_set():
                return
            label_widget.configure(image=photo, text='')
            label_widget._img = photo
            status_label.configure(text=f'Itération {iteration}/{N_ITERATIONS}')

        root.after(0, _update)

        # Calculer et mettre à jour la courbe d'énergie
        try:
            # On utilise le beta en cours (lu depuis le slider de façon dynamique)
            current_beta = float(slider_beta.get()) if 'slider_beta' in globals() else BETA
            e = convergence.compute_energy(current_labels, beta=current_beta)
            energy_history.append(e)
            
            # Mise à jour du graphique dans le thread principal de Tkinter (indispensable pour éviter les crashs)
            def _update_plot():
                if abort_event.is_set():
                    return
                energy_line.set_data(range(len(energy_history)), energy_history)
                energy_ax.relim()
                energy_ax.autoscale_view()
                energy_canvas.draw_idle()
                
            root.after(0, _update_plot)
        except Exception:
            pass

    return _cb


def trigger_live_gibbs():
    """Gère l'annulation du thread Gibbs en cours et lance un nouveau thread avec les paramètres actuels."""
    global gibbs_thread, abort_event, energy_history
    
    if current_image is None or initial_labels is None:
        return
        
    # Étape 1 : Signaler à l'ancien thread de s'arrêter
    abort_event.set()
    
    # Étape 2 : Vérification non-bloquante et asynchrone si le thread précédent tourne encore
    if gibbs_thread is not None and gibbs_thread.is_alive():
        # Reporter le démarrage de 50ms et quitter (évite de bloquer la boucle Tkinter et prévient les threads concurrents)
        root.after(50, trigger_live_gibbs)
        return
        
    # Étape 3 : Réinitialiser le signal d'annulation pour le nouveau run
    abort_event.clear()
    
    # Étape 4 : Réinitialiser la courbe d'énergie
    energy_history = []
    
    # Récupérer les valeurs actuelles des sliders
    beta = float(slider_beta.get()) if 'slider_beta' in globals() else BETA
    temperature = float(slider_t.get()) if 'slider_t' in globals() else TEMPERATURE
    
    # Callback thread-safe pour les mises à jour graphiques en direct
    cb = make_callback(root, label_image)
    
    # Étape 5 : Fonction d'exécution en tâche de fond (Thread)
    def _run_live():
        try:
            log(f'Lancement Gibbs live : beta={beta}, T={temperature}, iters={N_ITERATIONS}')
            # On utilise le mode vectorisé (vectorized=True) pour garantir le temps réel (15ms/iter)
            final, history = gibbs_sampler.run_gibbs(
                initial_labels.copy(),
                current_image,
                beta=beta,
                temperature=temperature,
                n_iter=N_ITERATIONS,
                callback=cb,
                vectorized=True,
                n_classes=N_CLASSES
            )
            
            if not abort_event.is_set():
                log('Gibbs terminé avec succès.')
                # Affichage de l'image finale
                rgb = _labels_to_rgb(final, N_CLASSES)
                im = Image.fromarray(rgb)
                photo = build_photo_image(im)
                
                def _final_update():
                    if abort_event.is_set():
                        return
                    label_image.configure(image=photo, text='')
                    label_image._img = photo
                    status_label.configure(text='Terminé')
                    
                root.after(0, _final_update)
        except Exception as e:
            if not abort_event.is_set():
                log(f'Erreur dans Gibbs live : {e}')
                root.after(0, lambda: messagebox.showerror('Erreur', f'Erreur Gibbs live : {e}'))

    # Démarrage du thread en tâche de fond (daemon pour quitter proprement avec l'app)
    gibbs_thread = threading.Thread(target=_run_live, daemon=True)
    gibbs_thread.start()


def on_slider_changed(val=None):
    """Fonction de rappel appelée lors du mouvement des curseurs Beta/Température.
    Implémente un debouncing de 200ms pour éviter de surcharger le processeur.
    """
    global debounce_timer
    
    # Annuler le timer en cours pour recommencer le debouncing
    if debounce_timer is not None:
        try:
            root.after_cancel(debounce_timer)
        except Exception:
            pass
            
    # Planifier le lancement dynamique approprié selon le workflow actif
    if active_workflow == 'A':
        if current_image is None or initial_labels is None:
            return
        debounce_timer = root.after(200, trigger_live_gibbs)
    elif active_workflow == 'B':
        if current_patch is None:
            return
        debounce_timer = root.after(200, trigger_live_synth)


def run_workflow_a():
    """Charge une image, initialise ses labels via K-means, et lance le premier run interactif."""
    global current_image, initial_labels, current_labels, active_workflow
    active_workflow = 'A'
    print('Démarrage du Workflow A...')
    
    # Sélection du fichier image
    path = filedialog.askopenfilename(title='Choisir une image', filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp')])
    if not path:
        log('Aucun fichier sélectionné — utilisation d\'une image simulée')
        H, W = IMAGE_SIZE
        image_array = np.random.randint(0, 256, (H, W, 3), dtype=np.uint8)
    else:
        try:
            image_array = image_loader.load_image(path)
        except Exception as e:
            messagebox.showerror('Erreur', f"Impossible de charger l'image : {e}")
            return

    # Afficher l'image originale dans la colonne de gauche
    try:
        show_energy_plot()
        im_orig = Image.fromarray(image_array)
        photo_orig = build_photo_image(im_orig)
        label_original.configure(image=photo_orig, text='')
        label_original._img = photo_orig
    except Exception:
        pass

    log('Image chargée et affichée (original).')

    # Initialisation des labels par K-means maison
    try:
        labels = initializer.init_labels(image_array, N_CLASSES)
    except Exception as e:
        messagebox.showerror('Erreur', f"Échec de l'initialisation : {e}")
        return

    # Sauvegarder les données de base pour permettre les mises à jour interactives en direct
    current_image = image_array
    initial_labels = labels
    current_labels = labels.copy()

    # Déclencher le premier traitement de segmentation
    trigger_live_gibbs()

def trigger_live_synth():
    """Gère la synthèse de texture dynamique (Workflow B) en tâche de fond de façon thread-safe."""
    global gibbs_thread, abort_event
    
    if current_patch is None:
        return
        
    # Étape 1 : Signaler à l'ancien thread de s'arrêter
    abort_event.set()
    
    # Étape 2 : Vérification non-bloquante et asynchrone si le thread précédent tourne encore
    if gibbs_thread is not None and gibbs_thread.is_alive():
        # Reporter le démarrage de 50ms et quitter (évite de bloquer la boucle Tkinter et prévient les threads concurrents)
        root.after(50, trigger_live_synth)
        return
        
    # Étape 3 : Réinitialiser l'événement d'annulation
    abort_event.clear()
    
    beta = float(slider_beta.get()) if 'slider_beta' in globals() else BETA
    temperature = float(slider_t.get()) if 'slider_t' in globals() else TEMPERATURE
    
    # Étape 4 : Thread de calcul de synthèse de texture
    def _run_synth():
        try:
            log('Démarrage de la synthèse...')
            
            # Capture temporaire de stdout
            import sys
            class _StdoutToLog:
                def write(self, s):
                    s = s.strip()
                    if s:
                        log(s)
                def flush(self):
                    pass
                    
            _old_stdout = sys.stdout
            sys.stdout = _StdoutToLog()
            try:
                # Appel au calcul mathématique de texture_synth
                labels = texture_synth.synthesize_texture(current_patch, output_size=IMAGE_SIZE, beta=beta, T=temperature)
            finally:
                sys.stdout = _old_stdout
                
            if abort_event.is_set():
                return
                
            log('Synthèse terminée.')

            # Convertir les labels en image avec les couleurs naturelles moyennes du patch d'origine
            rgb = _labels_to_natural_rgb(labels, current_patch)
            im = Image.fromarray(rgb)

            def _update_ui():
                try:
                    if abort_event.is_set():
                        return
                    # Création de PhotoImage et affichage (sur le thread principal de Tkinter)
                    photo = build_photo_image(im)
                    label_image.configure(image=photo, text='')
                    label_image._img = photo
                    status_label.configure(text='Terminé')
                except Exception as ex:
                    log(f"Erreur d'affichage de la texture: {ex}")

            root.after(0, _update_ui)
        except Exception as e:
            if not abort_event.is_set():
                log(f'Erreur pendant la synthèse: {e}')
                root.after(0, lambda: messagebox.showerror('Erreur', f"Erreur pendant la synthèse : {e}"))

    gibbs_thread = threading.Thread(target=_run_synth, daemon=True)
    gibbs_thread.start()


def run_workflow_b():
    """Charge un patch de texture et lance la synthèse en direct."""
    global active_workflow, current_patch
    print('Démarrage du Workflow B...')
    
    # Sélection du patch source
    path = filedialog.askopenfilename(title='Choisir un patch source', filetypes=[('Images', '*.png *.jpg *.jpeg *.bmp')])
    if not path:
        return

    try:
        patch = image_loader.load_image(path)
    except Exception as e:
        messagebox.showerror('Erreur', f"Impossible de charger le patch : {e}")
        return

    # Mettre à jour les variables d'état du workflow
    active_workflow = 'B'
    current_patch = patch

    # Afficher le patch original dans la colonne de gauche
    hide_energy_plot()
    try:
        im_orig = Image.fromarray(patch)
        photo_orig = build_photo_image(im_orig)
        label_original.configure(image=photo_orig, text='')
        label_original._img = photo_orig
    except Exception:
        pass

    log(f'Patch chargé: {os.path.basename(path)}')

    # Déclencher le premier traitement de synthèse de texture
    trigger_live_synth()


def ensure_icon_file(icon_path: str) -> str | None:
    try:
        os.makedirs(os.path.dirname(icon_path), exist_ok=True)
        if os.path.exists(icon_path):
            return icon_path
        png_data = base64.b64decode(ICON_PNG_B64)
        icon_dir = struct.pack('<3H', 0, 1, 1)
        entry = struct.pack('<BBBBHHII', 16, 16, 0, 0, 0, 32, len(png_data), 6 + 16)
        with open(icon_path, 'wb') as f:
            f.write(icon_dir)
            f.write(entry)
            f.write(png_data)
        return icon_path
    except Exception:
        return None


def apply_window_icon(root: tk.Tk):
    icon_path = ensure_icon_file(os.path.join(os.path.dirname(__file__), 'app_icon.ico'))
    if icon_path and os.name == 'nt':
        try:
            root.iconbitmap(icon_path)
        except Exception:
            pass
    try:
        icon_image = tk.PhotoImage(data=ICON_PNG_B64, format='png')
        root.iconphoto(True, icon_image)
        root._icon_image = icon_image
    except Exception:
        pass


def resize_for_display(image: Image.Image) -> Image.Image:
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return ImageOps.contain(image, IMAGE_DISPLAY_SIZE, Image.LANCZOS)


def build_photo_image(image: Image.Image) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(resize_for_display(image))


def show_energy_plot():
    try:
        global plot_frame, energy_line, energy_canvas
        energy_line.set_visible(True)
        plot_frame.grid()
        energy_canvas.draw_idle()
    except Exception:
        pass


def hide_energy_plot():
    try:
        global plot_frame, energy_line, energy_canvas
        energy_line.set_visible(False)
        plot_frame.grid_remove()
        energy_canvas.draw_idle()
    except Exception:
        pass


def create_ui():
    print('Creating UI...')
    global slider_beta, slider_t, label_image, label_original, status_label, root, log_widget
    global energy_fig, energy_ax, energy_canvas, energy_line, energy_history, plot_frame
    root = tk.Tk()
    root.title('Markov-Vision')
    root.geometry('860x760')
    root.minsize(680, 660)
    apply_window_icon(root)
    root.configure(bg=THEMES['dark']['root_bg'])
    root.deiconify()
    root.lift()
    root.attributes('-topmost', True)
    root.after(200, lambda: root.attributes('-topmost', False))
    # Start maximized so the UI fills the screen; user can restore if desired
    try:
        root.state('zoomed')
    except Exception:
        pass
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=0)
    root.rowconfigure(0, weight=1)

    # Scrollable container for the entire UI to avoid layout clipping on small windows
    canvas = tk.Canvas(root, bg=THEMES['dark']['root_bg'], highlightthickness=0)
    vscroll = tk.Scrollbar(root, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=vscroll.set)
    canvas.grid(row=0, column=0, sticky='nsew')
    vscroll.grid(row=0, column=1, sticky='ns')

    # Content frame placed inside canvas
    content = tk.Frame(canvas, bg=THEMES['dark']['main_bg'])
    window_id = canvas.create_window((0, 0), window=content, anchor='nw')

    # Allow the content frame to expand to fill the canvas window
    try:
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
    except Exception:
        pass

    def _on_content_config(event):
        canvas.configure(scrollregion=canvas.bbox('all'))

    content.bind('<Configure>', _on_content_config)

    # Ensure content width matches canvas width on resize
    def _on_canvas_config(event):
        try:
            canvas.itemconfig(window_id, width=event.width)
        except Exception:
            pass

    canvas.bind('<Configure>', _on_canvas_config)

    main_frame = tk.Frame(content, padx=12, pady=12, bg=THEMES['dark']['main_bg'])
    main_frame.grid(row=0, column=0, sticky='nsew')
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=5)
    main_frame.rowconfigure(1, weight=2)

    # Image frame: two columns -> original (left) and result (right)
    image_frame = tk.Frame(main_frame, bd=0, relief='flat', bg=THEMES['dark']['frame_bg'])
    image_frame.grid(row=0, column=0, sticky='nsew', padx=4, pady=(0, 10))
    image_frame.columnconfigure(0, weight=1)
    image_frame.columnconfigure(1, weight=1)
    image_frame.rowconfigure(0, weight=1)

    label_original = tk.Label(
        image_frame,
        text='Original',
        bg=THEMES['dark']['frame_bg'],
        fg=THEMES['dark']['fg'],
        anchor='center',
        justify='center',
        font=('Segoe UI', 12, 'bold'),
    )
    label_original.grid(row=0, column=0, sticky='nsew', padx=8, pady=8)

    label_image = tk.Label(
        image_frame,
        text=LANGUAGES['fr']['image'],
        bg=THEMES['dark']['frame_bg'],
        fg=THEMES['dark']['fg'],
        anchor='center',
        justify='center',
        font=('Segoe UI', 14, 'bold'),
    )
    label_image.grid(row=0, column=1, sticky='nsew', padx=8, pady=8)

    controls_frame = tk.Frame(main_frame, bg=THEMES['dark']['main_bg'])
    controls_frame.grid(row=1, column=0, sticky='nsew')
    controls_frame.columnconfigure(0, weight=1)

    toolbar = tk.Frame(controls_frame, bg=THEMES['dark']['main_bg'])
    toolbar.grid(row=0, column=0, sticky='ew', pady=(0, 10))
    toolbar.columnconfigure(0, weight=0)
    toolbar.columnconfigure(1, weight=1)
    toolbar.columnconfigure(2, weight=0)
    toolbar.columnconfigure(3, weight=0)

    icon_canvas = tk.Canvas(toolbar, width=26, height=26, highlightthickness=0, bg=THEMES['dark']['main_bg'])
    icon_canvas.grid(row=0, column=0, sticky='w', padx=(0, 8))
    icon_circle = icon_canvas.create_oval(2, 2, 24, 24, fill=THEMES['dark']['primary'], outline='')
    icon_canvas.create_text(13, 13, text='M', fill='white', font=('Segoe UI', 10, 'bold'))

    theme_toggle_btn = tk.Button(toolbar, text=LANGUAGES['fr']['switch_theme'], bd=0, padx=8, pady=6)
    theme_toggle_btn.grid(row=0, column=1, sticky='w', padx=(0, 8))

    lang_toggle_btn = tk.Button(toolbar, text=LANGUAGES['fr']['switch_lang'], bd=0, padx=8, pady=6)
    lang_toggle_btn.grid(row=0, column=2, sticky='e')

    logs_toggle_btn = tk.Button(toolbar, text=LANGUAGES['fr']['toggle_logs'], bd=0, padx=8, pady=6)
    logs_toggle_btn.grid(row=0, column=3, sticky='e', padx=(8,0))

    # Matplotlib plot area for energy convergence (above sliders)
    plot_frame = tk.Frame(controls_frame, bg=THEMES['dark']['main_bg'])
    plot_frame.grid(row=1, column=0, sticky='ew', pady=(0, 12))
    plot_frame.columnconfigure(0, weight=1)

    # Create empty figure and canvas
    energy_history = []
    energy_fig = Figure(figsize=(6, 1.0), dpi=100, facecolor=THEMES['dark']['main_bg'], constrained_layout=True)
    energy_ax = energy_fig.add_subplot(111)
    energy_ax.set_facecolor(THEMES['dark']['frame_bg'])
    energy_ax.plot([], [])
    energy_ax.set_title('Convergence (Énergie)', color=THEMES['dark']['fg'])
    energy_ax.tick_params(colors=THEMES['dark']['fg'])
    energy_ax.spines['bottom'].set_color('#444')
    energy_ax.spines['left'].set_color('#444')
    energy_line, = energy_ax.plot([], [], color='#3c82f6', marker='.', linewidth=2)

    energy_canvas = FigureCanvasTkAgg(energy_fig, master=plot_frame)
    energy_canvas.draw()
    energy_canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')
    plot_frame.rowconfigure(0, weight=1)
    # plot_frame is hidden by default; allow it to expand when shown
    plot_frame.grid_remove()

    # Resize the matplotlib figure when the plot_frame size changes so title and axes scale
    _in_resize = False
    def _on_plot_resize(event):
        nonlocal _in_resize
        if _in_resize:
            return
        try:
            _in_resize = True
            dpi = energy_fig.dpi or 100
            w_in = max(1.0, event.width / dpi)
            h_in = max(0.5, event.height / dpi)
            energy_fig.set_size_inches(w_in, h_in)
            # nudge title to avoid clipping and recompute layout
            try:
                energy_ax.title.set_y(1.05)
            except Exception:
                pass
            energy_canvas.draw_idle()
        except Exception:
            pass
        finally:
            _in_resize = False

    plot_frame.bind('<Configure>', _on_plot_resize)

    # Cadre d'entête horizontal pour Beta avec modification de la plage max
    beta_header_frame = tk.Frame(controls_frame, bg=THEMES['dark']['main_bg'])
    beta_header_frame.grid(row=2, column=0, sticky='ew', pady=(0, 4))
    beta_header_frame.columnconfigure(0, weight=1)
    beta_header_frame.columnconfigure(1, weight=0)
    beta_header_frame.columnconfigure(2, weight=0)

    label_beta = tk.Label(beta_header_frame, text=LANGUAGES['fr']['beta'], anchor='w', fg=THEMES['dark']['fg'], bg=THEMES['dark']['main_bg'], font=CONTROL_FONT_BOLD)
    label_beta.grid(row=0, column=0, sticky='w')

    label_beta_max = tk.Label(beta_header_frame, text="Max:", fg=THEMES['dark']['fg'], bg=THEMES['dark']['main_bg'], font=CONTROL_FONT)
    label_beta_max.grid(row=0, column=1, sticky='e', padx=(8, 4))

    entry_beta_max = tk.Entry(beta_header_frame, width=5, font=CONTROL_FONT, bg=THEMES['dark']['frame_bg'], fg=THEMES['dark']['fg'], insertbackground=THEMES['dark']['fg'], bd=1, relief='flat')
    entry_beta_max.insert(0, "5.0")
    entry_beta_max.grid(row=0, column=2, sticky='e')

    slider_beta = tk.Scale(
        controls_frame,
        from_=0.0,
        to=5.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        length=SLIDER_LENGTH,
        font=CONTROL_FONT,
        bg=THEMES['dark']['main_bg'],
        fg=THEMES['dark']['fg'],
        troughcolor=THEMES['dark']['trough'],
        highlightthickness=0,
        bd=0,
        command=on_slider_changed
    )
    slider_beta.set(1.5)
    slider_beta.grid(row=3, column=0, sticky='ew')

    def update_beta_range(event=None):
        try:
            val = float(entry_beta_max.get())
            if val > 0:
                slider_beta.configure(to=val)
        except ValueError:
            pass

    entry_beta_max.bind('<Return>', update_beta_range)
    entry_beta_max.bind('<FocusOut>', update_beta_range)

    # Cadre d'entête horizontal pour T avec modification de la plage max
    t_header_frame = tk.Frame(controls_frame, bg=THEMES['dark']['main_bg'])
    t_header_frame.grid(row=4, column=0, sticky='ew', pady=(12, 4))
    t_header_frame.columnconfigure(0, weight=1)
    t_header_frame.columnconfigure(1, weight=0)
    t_header_frame.columnconfigure(2, weight=0)

    label_t = tk.Label(t_header_frame, text=LANGUAGES['fr']['temperature'], anchor='w', fg=THEMES['dark']['fg'], bg=THEMES['dark']['main_bg'], font=CONTROL_FONT_BOLD)
    label_t.grid(row=0, column=0, sticky='w')

    label_t_max = tk.Label(t_header_frame, text="Max:", fg=THEMES['dark']['fg'], bg=THEMES['dark']['main_bg'], font=CONTROL_FONT)
    label_t_max.grid(row=0, column=1, sticky='e', padx=(8, 4))

    entry_t_max = tk.Entry(t_header_frame, width=5, font=CONTROL_FONT, bg=THEMES['dark']['frame_bg'], fg=THEMES['dark']['fg'], insertbackground=THEMES['dark']['fg'], bd=1, relief='flat')
    entry_t_max.insert(0, "5.0")
    entry_t_max.grid(row=0, column=2, sticky='e')

    slider_t = tk.Scale(
        controls_frame,
        from_=0.1,
        to=5.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        length=SLIDER_LENGTH,
        font=CONTROL_FONT,
        bg=THEMES['dark']['main_bg'],
        fg=THEMES['dark']['fg'],
        troughcolor=THEMES['dark']['trough'],
        highlightthickness=0,
        bd=0,
        command=on_slider_changed
    )
    slider_t.set(1.0)
    slider_t.grid(row=5, column=0, sticky='ew')

    def update_t_range(event=None):
        try:
            val = float(entry_t_max.get())
            if val > 0:
                slider_t.configure(to=val)
        except ValueError:
            pass

    entry_t_max.bind('<Return>', update_t_range)
    entry_t_max.bind('<FocusOut>', update_t_range)

    button_frame = tk.Frame(controls_frame, bg=THEMES['dark']['main_bg'])
    button_frame.grid(row=6, column=0, sticky='ew', pady=(16, 5))
    button_frame.columnconfigure(0, weight=1)
    button_frame.columnconfigure(1, weight=1)

    btn_font_a = tkfont.Font(family='Segoe UI', size=10, weight='bold')
    btn_font_b = tkfont.Font(family='Segoe UI', size=10, weight='bold')

    btn_a = tk.Button(
        button_frame,
        text=LANGUAGES['fr']['workflow_a'],
        command=run_workflow_a,
        bg=THEMES['dark']['primary'],
        fg='#ffffff',
        activebackground=THEMES['dark']['primary_hover'],
        activeforeground='#ffffff',
        bd=0,
        padx=BUTTON_PAD_X,
        pady=BUTTON_PAD_Y,
        font=btn_font_a,
    )
    btn_a.grid(row=0, column=0, sticky='ew', padx=(0, 4))

    btn_b = tk.Button(
        button_frame,
        text=LANGUAGES['fr']['workflow_b'],
        command=run_workflow_b,
        bg=THEMES['dark']['secondary'],
        fg='#ffffff',
        activebackground=THEMES['dark']['secondary_hover'],
        activeforeground='#ffffff',
        bd=0,
        padx=BUTTON_PAD_X,
        pady=BUTTON_PAD_Y,
        font=btn_font_b,
    )
    btn_b.grid(row=0, column=1, sticky='ew')

    # Zone de logs (multi-lignes) placée au-dessus du libellé de statut
    log_frame = tk.Frame(content, bg=THEMES['dark']['root_bg'])
    log_frame.grid(row=1, column=0, sticky='nsew', padx=12, pady=(6, 6))
    log_frame.columnconfigure(0, weight=1)
    # Fixer une hauteur réduite pour éviter d'écraser les contrôles
    log_frame.configure(height=120)
    log_frame.grid_propagate(False)

    log_scroll = tk.Scrollbar(log_frame)
    log_scroll.grid(row=0, column=1, sticky='ns')

    log_widget = tk.Text(log_frame, height=4, bg=THEMES['dark']['frame_bg'], fg=THEMES['dark']['fg'], wrap='word', yscrollcommand=log_scroll.set)
    log_widget.grid(row=0, column=0, sticky='nsew')
    log_widget.configure(state='disabled', padx=6, pady=6)
    log_scroll.config(command=log_widget.yview)

    status_label = tk.Label(content, text=LANGUAGES['fr']['status'], anchor='w', fg=THEMES['dark']['status_fg'], bg=THEMES['dark']['root_bg'], font=('Segoe UI', 9))
    status_label.grid(row=2, column=0, sticky='ew', padx=12, pady=(0, 12))

    # log_widget est déclaré global en tête de la fonction

    current_theme = {'name': 'dark'}
    current_lang = {'code': 'fr'}

    def apply_theme(name: str):
        t = THEMES[name]
        spine_color = '#444' if name == 'dark' else '#888'
        root.configure(bg=t['root_bg'])
        main_frame.configure(bg=t['main_bg'])
        image_frame.configure(bg=t['frame_bg'])
        label_image.configure(bg=t['frame_bg'], fg=t['fg'])
        controls_frame.configure(bg=t['main_bg'])
        toolbar.configure(bg=t['main_bg'])
        icon_canvas.configure(bg=t['main_bg'])
        beta_header_frame.configure(bg=t['main_bg'])
        label_beta.configure(bg=t['main_bg'], fg=t['fg'])
        label_beta_max.configure(bg=t['main_bg'], fg=t['fg'])
        entry_beta_max.configure(bg=t['frame_bg'], fg=t['fg'], insertbackground=t['fg'])

        t_header_frame.configure(bg=t['main_bg'])
        label_t.configure(bg=t['main_bg'], fg=t['fg'])
        label_t_max.configure(bg=t['main_bg'], fg=t['fg'])
        entry_t_max.configure(bg=t['frame_bg'], fg=t['fg'], insertbackground=t['fg'])

        slider_beta.configure(bg=t['main_bg'], fg=t['fg'], troughcolor=t['trough'])
        slider_t.configure(bg=t['main_bg'], fg=t['fg'], troughcolor=t['trough'])
        btn_a.configure(bg=t['primary'], activebackground=t['primary_hover'])
        btn_b.configure(bg=t['secondary'], activebackground=t['secondary_hover'])
        status_label.configure(bg=t['root_bg'], fg=t['status_fg'])
        icon_canvas.itemconfig(icon_circle, fill=t['primary'])
        energy_fig.set_facecolor(t['main_bg'])
        try:
            logs_toggle_btn.configure(bg=t['main_bg'], fg=t['fg'])
        except Exception:
            pass
        energy_ax.set_facecolor(t['frame_bg'])
        energy_ax.set_title(LANGUAGES[current_lang['code']]['plot_title'], color=t['fg'])
        energy_ax.tick_params(colors=t['fg'])
        energy_ax.spines['bottom'].set_color(spine_color)
        energy_ax.spines['left'].set_color(spine_color)
        energy_canvas.draw_idle()

    def set_language(code: str):
        lang = LANGUAGES[code]
        label_beta.configure(text=lang['beta'])
        label_t.configure(text=lang['temperature'])
        btn_a.configure(text=lang['workflow_a'])
        btn_b.configure(text=lang['workflow_b'])
        status_label.configure(text=lang['status'])
        label_image.configure(text=lang['image'])
        theme_toggle_btn.configure(text=lang['switch_theme'])
        lang_toggle_btn.configure(text=lang['switch_lang'])
        energy_ax.set_title(lang['plot_title'], color=THEMES[current_theme['name']]['fg'])
        try:
            logs_toggle_btn.configure(text=lang['toggle_logs'])
        except Exception:
            pass
        energy_canvas.draw_idle()

    def toggle_logs():
        try:
            if log_frame.winfo_viewable():
                log_frame.grid_remove()
                logs_toggle_btn.configure(text=LANGUAGES[current_lang['code']]['show_logs'])
            else:
                log_frame.grid()
                logs_toggle_btn.configure(text=LANGUAGES[current_lang['code']]['hide_logs'])
        except Exception:
            pass
    def toggle_theme():
        current_theme['name'] = 'light' if current_theme['name'] == 'dark' else 'dark'
        apply_theme(current_theme['name'])

    def toggle_lang():
        current_lang['code'] = 'en' if current_lang['code'] == 'fr' else 'fr'
        set_language(current_lang['code'])

    def on_enter_a(event):
        btn_a.configure(bg=THEMES[current_theme['name']]['primary_hover'], relief='raised')

    def on_leave_a(event):
        btn_a.configure(bg=THEMES[current_theme['name']]['primary'], relief='flat')

    def on_enter_b(event):
        btn_b.configure(bg=THEMES[current_theme['name']]['secondary_hover'], relief='raised')

    def on_leave_b(event):
        btn_b.configure(bg=THEMES[current_theme['name']]['secondary'], relief='flat')

    btn_a.bind('<Enter>', on_enter_a)
    btn_a.bind('<Leave>', on_leave_a)
    btn_b.bind('<Enter>', on_enter_b)
    btn_b.bind('<Leave>', on_leave_b)
    theme_toggle_btn.configure(command=toggle_theme)
    lang_toggle_btn.configure(command=toggle_lang)
    try:
        logs_toggle_btn.configure(command=toggle_logs)
    except Exception:
        pass

    apply_theme(current_theme['name'])
    set_language(current_lang['code'])
    print('UI ready, entering mainloop')
    root.mainloop()


if __name__ == '__main__':
    print('Starting UI...')
    try:
        create_ui()
        print('UI exited cleanly')
    except Exception as error:
        print('UI failed:', error)
        import traceback
        traceback.print_exc()
