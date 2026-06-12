import base64
import os
import struct
import tkinter as tk
import tkinter.font as tkfont

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


def run_workflow_a():
    print('Démarrage du Workflow A...')


def run_workflow_b():
    print('Démarrage du Workflow B...')


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


def create_ui():
    print('Creating UI...')
    root = tk.Tk()
    root.title('Markov-Vision')
    root.geometry('720x820')
    root.minsize(560, 640)
    apply_window_icon(root)
    root.configure(bg=THEMES['dark']['root_bg'])
    root.deiconify()
    root.lift()
    root.attributes('-topmost', True)
    root.after(200, lambda: root.attributes('-topmost', False))
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    main_frame = tk.Frame(root, padx=16, pady=16, bg=THEMES['dark']['main_bg'])
    main_frame.grid(row=0, column=0, sticky='nsew')
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(0, weight=3)
    main_frame.rowconfigure(1, weight=2)

    image_frame = tk.Frame(main_frame, bd=0, relief='flat', bg=THEMES['dark']['frame_bg'])
    image_frame.grid(row=0, column=0, sticky='nsew', padx=4, pady=(0, 10))
    image_frame.columnconfigure(0, weight=1)
    image_frame.rowconfigure(0, weight=1)

    label_image = tk.Label(
        image_frame,
        text=LANGUAGES['fr']['image'],
        bg=THEMES['dark']['frame_bg'],
        fg=THEMES['dark']['fg'],
        anchor='center',
        justify='center',
        font=('Segoe UI', 14, 'bold'),
    )
    label_image.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)

    controls_frame = tk.Frame(main_frame, bg=THEMES['dark']['main_bg'])
    controls_frame.grid(row=1, column=0, sticky='nsew')
    controls_frame.columnconfigure(0, weight=1)

    toolbar = tk.Frame(controls_frame, bg=THEMES['dark']['main_bg'])
    toolbar.grid(row=0, column=0, sticky='ew', pady=(0, 10))
    toolbar.columnconfigure(0, weight=0)
    toolbar.columnconfigure(1, weight=1)
    toolbar.columnconfigure(2, weight=0)

    icon_canvas = tk.Canvas(toolbar, width=26, height=26, highlightthickness=0, bg=THEMES['dark']['main_bg'])
    icon_canvas.grid(row=0, column=0, sticky='w', padx=(0, 8))
    icon_circle = icon_canvas.create_oval(2, 2, 24, 24, fill=THEMES['dark']['primary'], outline='')
    icon_canvas.create_text(13, 13, text='M', fill='white', font=('Segoe UI', 10, 'bold'))

    theme_toggle_btn = tk.Button(toolbar, text=LANGUAGES['fr']['switch_theme'], bd=0, padx=8, pady=6)
    theme_toggle_btn.grid(row=0, column=1, sticky='w', padx=(0, 8))

    lang_toggle_btn = tk.Button(toolbar, text=LANGUAGES['fr']['switch_lang'], bd=0, padx=8, pady=6)
    lang_toggle_btn.grid(row=0, column=2, sticky='e')

    label_beta = tk.Label(controls_frame, text=LANGUAGES['fr']['beta'], anchor='w', fg=THEMES['dark']['fg'], bg=THEMES['dark']['main_bg'], font=('Segoe UI', 11, 'bold'))
    label_beta.grid(row=1, column=0, sticky='ew', pady=(0, 4))

    slider_beta = tk.Scale(
        controls_frame,
        from_=0.0,
        to=5.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        bg=THEMES['dark']['main_bg'],
        fg=THEMES['dark']['fg'],
        troughcolor=THEMES['dark']['trough'],
        highlightthickness=0,
    )
    slider_beta.set(1.5)
    slider_beta.grid(row=2, column=0, sticky='ew')

    label_t = tk.Label(controls_frame, text=LANGUAGES['fr']['temperature'], anchor='w', fg=THEMES['dark']['fg'], bg=THEMES['dark']['main_bg'], font=('Segoe UI', 11, 'bold'))
    label_t.grid(row=3, column=0, sticky='ew', pady=(12, 4))

    slider_t = tk.Scale(
        controls_frame,
        from_=0.1,
        to=5.0,
        resolution=0.1,
        orient=tk.HORIZONTAL,
        bg=THEMES['dark']['main_bg'],
        fg=THEMES['dark']['fg'],
        troughcolor=THEMES['dark']['trough'],
        highlightthickness=0,
    )
    slider_t.set(1.0)
    slider_t.grid(row=4, column=0, sticky='ew')

    btn_font_a = tkfont.Font(family='Segoe UI', size=11, weight='bold')
    btn_font_b = tkfont.Font(family='Segoe UI', size=11, weight='bold')

    btn_a = tk.Button(
        controls_frame,
        text=LANGUAGES['fr']['workflow_a'],
        command=run_workflow_a,
        bg=THEMES['dark']['primary'],
        fg='#ffffff',
        activebackground=THEMES['dark']['primary_hover'],
        activeforeground='#ffffff',
        bd=0,
        padx=10,
        pady=10,
        font=btn_font_a,
    )
    btn_a.grid(row=5, column=0, sticky='ew', pady=(16, 5))

    btn_b = tk.Button(
        controls_frame,
        text=LANGUAGES['fr']['workflow_b'],
        command=run_workflow_b,
        bg=THEMES['dark']['secondary'],
        fg='#ffffff',
        activebackground=THEMES['dark']['secondary_hover'],
        activeforeground='#ffffff',
        bd=0,
        padx=10,
        pady=10,
        font=btn_font_b,
    )
    btn_b.grid(row=6, column=0, sticky='ew', pady=(0, 5))

    status_label = tk.Label(root, text=LANGUAGES['fr']['status'], anchor='w', fg=THEMES['dark']['status_fg'], bg=THEMES['dark']['root_bg'], font=('Segoe UI', 9))
    status_label.grid(row=1, column=0, sticky='ew', padx=12, pady=(0, 12))

    current_theme = {'name': 'dark'}
    current_lang = {'code': 'fr'}

    def apply_theme(name: str):
        t = THEMES[name]
        root.configure(bg=t['root_bg'])
        main_frame.configure(bg=t['main_bg'])
        image_frame.configure(bg=t['frame_bg'])
        label_image.configure(bg=t['frame_bg'], fg=t['fg'])
        controls_frame.configure(bg=t['main_bg'])
        toolbar.configure(bg=t['main_bg'])
        icon_canvas.configure(bg=t['main_bg'])
        label_beta.configure(bg=t['main_bg'], fg=t['fg'])
        label_t.configure(bg=t['main_bg'], fg=t['fg'])
        slider_beta.configure(bg=t['main_bg'], fg=t['fg'], troughcolor=t['trough'])
        slider_t.configure(bg=t['main_bg'], fg=t['fg'], troughcolor=t['trough'])
        btn_a.configure(bg=t['primary'], activebackground=t['primary_hover'])
        btn_b.configure(bg=t['secondary'], activebackground=t['secondary_hover'])
        status_label.configure(bg=t['root_bg'], fg=t['status_fg'])
        icon_canvas.itemconfig(icon_circle, fill=t['primary'])

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
