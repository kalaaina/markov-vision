import tkinter as tk

# Ces fonctions s'exécuteront quand on cliquera sur les boutons. 
# Pour l'instant, elles affichent juste un texte dans le terminal (pas de vraie logique).
def run_workflow_a():
    print("Démarrage du Workflow A...")

def run_workflow_b():
    print("Démarrage du Workflow B...")

def create_ui():
    # 1. Création de la fenêtre principale
    root = tk.Tk()
    root.title("Markov-Vision")
    root.geometry("400x500") # Largeur x Hauteur

    # 2. Espace temporaire pour l'affichage de l'image (Placeholder)
    label_image = tk.Label(root, text="[ L'image s'affichera ici ]", bg="lightgray", width=40, height=15)
    label_image.pack(pady=20) # pady ajoute de l'espace au-dessus et en dessous

    # 3. Curseur (Slider) pour le paramètre Beta
    label_beta = tk.Label(root, text="Paramètre Beta (Cohésion):")
    label_beta.pack()
    slider_beta = tk.Scale(root, from_=0.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL)
    slider_beta.set(1.5) # Valeur par défaut
    slider_beta.pack()

    # 4. Curseur (Slider) pour le paramètre T (Température)
    label_t = tk.Label(root, text="Paramètre T (Température):")
    label_t.pack()
    slider_t = tk.Scale(root, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL)
    slider_t.set(1.0) # Valeur par défaut
    slider_t.pack(pady=10)

    # 5. Les deux boutons Workflow A et B
    btn_a = tk.Button(root, text="Lancer Workflow A", command=run_workflow_a)
    btn_a.pack(pady=5)

    btn_b = tk.Button(root, text="Lancer Workflow B", command=run_workflow_b)
    btn_b.pack(pady=5)

    # 6. Démarrage de la boucle de la fenêtre (pour qu'elle reste ouverte)
    root.mainloop()

# Bloc pour tester ton code indépendamment des autres
if __name__ == '__main__':
    create_ui()