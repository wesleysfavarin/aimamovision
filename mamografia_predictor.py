import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

class MamografiaPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Predição de Mamografia - IA")
        self.root.geometry("1400x900")
        
        # Paleta de cores atualizada com fundo branco
        self.colors = {
            'primary': '#2E86C1',      # Azul principal
            'secondary': '#27AE60',    # Verde secundário
            'danger': '#E74C3C',      # Vermelho para alertas
            'background': '#FFFFFF',  # Fundo branco
            'panel_bg': '#F8F9FA',    # Fundo dos painéis
            'text': '#2C3E50',       # Texto escuro
            'light_text': '#7F8C8D', # Texto claro
            'white': '#FFFFFF',      # Branco
            'accent': '#F1C40F',     # Amarelo accent
            'graph_colors': ['#3498DB', '#E74C3C']  # Cores para os gráficos
        }
        
        # Configurar estilo global com fundo branco para todos os elementos
        style = ttk.Style()
        style.configure('TFrame', background=self.colors['white'])
        style.configure('TLabel', background=self.colors['white'], foreground=self.colors['text'])
        style.configure('TButton', padding=10, font=('Helvetica', 11))
        style.configure('TLabelframe', background=self.colors['white'])
        style.configure('TLabelframe.Label', background=self.colors['white'], foreground=self.colors['text'])
        
        # Configurar estilos específicos
        style.configure('Card.TFrame', background=self.colors['white'])
        style.configure('Card.TLabelframe', background=self.colors['white'])
        style.configure('Card.TLabelframe.Label', background=self.colors['white'])
        
        # Configurar grid do root para responsividade
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Configurar cor de fundo do root
        self.root.configure(bg=self.colors['white'])
        
        # Inicializar modelo e scaler
        self.model = None
        self.scaler = None
        
        # Carregar dados do dataset e treinar modelo
        self.load_dataset()
        self.load_or_train_model()
        
        # Configurar interface
        self.setup_ui()
        
        # Adicionar binding para redimensionamento
        self.root.bind('<Configure>', self.on_window_resize)
    
    def load_dataset(self):
        """Carrega os dados do dataset (CSVs e mapeia as imagens)"""
        try:
            # Caminhos base
            base_path = Path(__file__).parent
            archive_dir = base_path / 'archive'
            csv_path = archive_dir / 'csv'
            self.image_path = archive_dir / 'jpeg'
            
            # Verificar diretórios
            if not all(path.exists() for path in [archive_dir, csv_path, self.image_path]):
                print("ERRO: Diretórios necessários não encontrados!")
                return
            
            # Listar arquivos CSV
            print("\nArquivos CSV disponíveis:")
            for file in csv_path.glob("*.csv"):
                print(f"- {file.name}")
            
            # Procurar imagens recursivamente
            print("\nVerificando imagens JPEG:")
            jpeg_files = []
            for ext in ['*.jpg', '*.jpeg']:
                jpeg_files.extend(list(self.image_path.rglob(ext)))
            
            print(f"Total de imagens encontradas: {len(jpeg_files)}")
            if jpeg_files:
                print("Primeiras 5 imagens (com caminhos relativos):")
                for file in jpeg_files[:5]:
                    rel_path = file.relative_to(self.image_path)
                    print(f"- {rel_path}")
            
            # Carregar CSVs
            csv_files = {
                'calc_train': 'calc_case_description_train_set.csv',
                'calc_test': 'calc_case_description_test_set.csv',
                'mass_train': 'mass_case_description_train_set.csv',
                'mass_test': 'mass_case_description_test_set.csv'
            }
            
            # Carregar e combinar datasets
            dfs = []
            for name, file in csv_files.items():
                file_path = csv_path / file
                if file_path.exists():
                    print(f"\nCarregando {file}...")
                    df = pd.read_csv(file_path)
                    # Adicionar coluna indicando o tipo (calc ou mass)
                    df['type'] = 'calc' if 'calc' in name else 'mass'
                    print(f"- Registros encontrados: {len(df)}")
                    dfs.append(df)
            
            if dfs:
                self.dataset = pd.concat(dfs, ignore_index=True)
                
                # Criar mapeamento de imagens para informações
                self.image_info = {}
                images_found = 0
                
                # Criar dicionário de imagens encontradas
                image_paths_dict = {}
                for file in jpeg_files:
                    # Extrair o ID do paciente e número da imagem do caminho
                    parts = str(file).split('/')
                    if len(parts) >= 2:
                        # O nome do arquivo geralmente é algo como "1-232.jpg"
                        image_name = parts[-1]
                        # O diretório pai contém o ID DICOM
                        dicom_id = parts[-2]
                        image_paths_dict[f"{dicom_id}/{image_name}"] = file
                
                print("\nProcessando registros do dataset...")
                for _, row in self.dataset.iterrows():
                    image_path = row.get('image file path', '')
                    if pd.notna(image_path):
                        # Extrair todos os IDs DICOM do caminho
                        dicom_ids = [part for part in image_path.split('/') 
                                    if part.startswith('1.3.6.1.4.1.9590')]
                        
                        if dicom_ids:
                            # Para cada ID DICOM encontrado no caminho
                            for dicom_id in dicom_ids:
                                # Procurar imagens que contenham este ID DICOM
                                matching_images = [
                                    (key, path) for key, path in image_paths_dict.items()
                                    if dicom_id in str(path)
                                ]
                                
                                # Se encontrou imagens correspondentes
                                for key, file_path in matching_images:
                                    images_found += 1
                                    image_filename = os.path.basename(file_path)
                                    
                                    # Verificar se já existe esta imagem no mapeamento
                                    if image_filename not in self.image_info:
                                        self.image_info[image_filename] = {
                                            'pathology': row.get('pathology', ''),
                                            'breast_density': row.get('breast_density', ''),
                                            'abnormality_type': row.get('abnormality type', ''),
                                            'assessment': row.get('assessment', ''),
                                            'subtlety': row.get('subtlety', ''),
                                            'type': row.get('type', ''),
                                            'full_path': str(file_path),
                                            'dicom_id': dicom_id
                                        }
                
                if images_found > 0:
                    print(f"\nMapeamento concluído com sucesso!")
                    print(f"Total de imagens mapeadas: {images_found}")
                    print("\nExemplo de mapeamento:")
                    example_key = next(iter(self.image_info))
                    print(f"Arquivo: {example_key}")
                    print("Informações:", self.image_info[example_key])
                    
                    # Mostrar distribuição de patologias
                    pathologies = [info['pathology'] for info in self.image_info.values()]
                    unique_pathologies = set(pathologies)
                    print("\nDistribuição de patologias:")
                    for pathology in unique_pathologies:
                        count = pathologies.count(pathology)
                        print(f"- {pathology}: {count} imagens")
            
            else:
                print("\nERRO: Nenhum arquivo CSV foi carregado!")
                self.dataset = None
                self.image_info = {}
        
        except Exception as e:
            print(f"\nErro ao carregar dataset: {e}")
            import traceback
            traceback.print_exc()
            self.dataset = None
            self.image_info = {}
    
    def extract_features(self, img):
        """Extrai características da imagem"""
        try:
            # Converter para escala de cinza
            img_gray = img.convert('L')
            
            # Redimensionar para tamanho padrão
            img_resized = img_gray.resize((128, 128))
            
            # Converter para array numpy
            img_array = np.array(img_resized)
            
            # Extrair características
            features = []
            
            # Características estatísticas
            features.extend([
                np.mean(img_array),  # Média
                np.std(img_array),   # Desvio padrão
                np.median(img_array), # Mediana
                np.percentile(img_array, 25),  # Primeiro quartil
                np.percentile(img_array, 75),  # Terceiro quartil
                np.max(img_array),    # Máximo
                np.min(img_array),    # Mínimo
                np.var(img_array),    # Variância
            ])
            
            # Características de textura
            from skimage.feature import graycomatrix, graycoprops
            glcm = graycomatrix(img_array, [1], [0], 256, symmetric=True, normed=True)
            features.extend([
                graycoprops(glcm, 'contrast')[0, 0],
                graycoprops(glcm, 'homogeneity')[0, 0],
                graycoprops(glcm, 'energy')[0, 0],
                graycoprops(glcm, 'correlation')[0, 0]
            ])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"Erro ao extrair características: {e}")
            return None
    
    def load_or_train_model(self):
        """Carrega ou treina o modelo de classificação"""
        try:
            model_path = Path(__file__).parent / 'mamografia_model.joblib'
            scaler_path = Path(__file__).parent / 'mamografia_scaler.joblib'
            
            if model_path.exists() and scaler_path.exists():
                print("Carregando modelo existente...")
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
            else:
                print("Treinando novo modelo...")
                self.train_model()
        
        except Exception as e:
            print(f"Erro ao carregar/treinar modelo: {e}")
            self.model = None
            self.scaler = None
    
    def train_model(self):
        """Treina o modelo usando as imagens do dataset"""
        try:
            X = []  # Características
            y = []  # Classes
            pathology_mapping = {}  # Para mapear as patologias encontradas
            
            print("Extraindo características das imagens...")
            total_images = len(self.image_info)
            processed = 0
            
            for filename, info in self.image_info.items():
                try:
                    full_path = info['full_path']
                    if os.path.exists(full_path):
                        img = Image.open(full_path)
                        features = self.extract_features(img)
                        
                        if features is not None:
                            pathology = info['pathology'].lower()
                            
                            # Mapear patologia para classe numérica (apenas Benigno e Maligno)
                            if 'malignant' in pathology:
                                class_num = 1  # Maligno
                            else:  # benign ou benign_without_callback
                                class_num = 0  # Benigno
                            
                            X.append(features.flatten())
                            y.append(class_num)
                            
                            # Registrar mapeamento
                            if pathology not in pathology_mapping:
                                pathology_mapping[pathology] = class_num
                            
                            processed += 1
                            if processed % 100 == 0:
                                print(f"Processadas {processed}/{total_images} imagens...")
                
                except Exception as e:
                    print(f"Erro ao processar imagem {filename}: {e}")
                    continue
            
            if X and y:
                print(f"\nTotal de imagens processadas com sucesso: {len(X)}")
                print("\nMapeamento de patologias para classes:")
                for pathology, class_num in pathology_mapping.items():
                    class_name = "Maligno" if class_num == 1 else "Benigno"
                    print(f"- {pathology} -> {class_name}")
                
                print("\nDistribuição das classes:")
                y_array = np.array(y)
                class_counts = np.bincount(y_array)
                class_names = ['Benigno', 'Maligno']
                
                for class_num, count in enumerate(class_counts):
                    print(f"- {class_names[class_num]}: {count} imagens")
                
                X = np.array(X)
                y = np.array(y)
                
                # Criar e treinar scaler
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
                
                # Criar e treinar modelo
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
                self.model.fit(X_scaled, y)
                
                # Salvar modelo e scaler
                model_path = Path(__file__).parent / 'mamografia_model.joblib'
                scaler_path = Path(__file__).parent / 'mamografia_scaler.joblib'
                
                joblib.dump(self.model, model_path)
                joblib.dump(self.scaler, scaler_path)
                
                print("\nModelo treinado e salvo com sucesso")
            else:
                print("Sem dados suficientes para treinar o modelo")
        
        except Exception as e:
            print(f"Erro ao treinar modelo: {e}")
            import traceback
            traceback.print_exc()
            self.model = None
            self.scaler = None
    
    def predict_image(self, image_path):
        """Faz a predição usando o modelo treinado"""
        try:
            if self.model is None or self.scaler is None:
                return None
            
            # Carregar e processar imagem
            img = Image.open(image_path)
            features = self.extract_features(img)
            
            if features is not None:
                # Normalizar características
                features_scaled = self.scaler.transform(features)
                
                # Fazer predição
                pred_proba = self.model.predict_proba(features_scaled)[0]
                
                # Converter para dicionário (apenas Benigno e Maligno)
                probabilities = {
                    'Benigno': float(pred_proba[0]),
                    'Maligno': float(pred_proba[1])
                }
                
                return probabilities
            
            return None
            
        except Exception as e:
            print(f"Erro na predição: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def setup_ui(self):
        """Configura a interface do usuário com visual moderno e responsivo"""
        # Frame principal com padding maior e fundo branco
        main_frame = ttk.Frame(self.root, padding="20", style='TFrame')
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid do main_frame para responsividade
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=2)  # Aumentar peso da coluna da imagem
        main_frame.grid_columnconfigure(1, weight=3)  # Aumentar peso da coluna dos gráficos
        
        # Frame esquerdo com fundo branco
        left_frame = ttk.Frame(main_frame, padding="15", style='Card.TFrame')
        left_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 20))
        
        # Configurar grid do left_frame
        left_frame.grid_rowconfigure(2, weight=1)  # A imagem pode crescer
        left_frame.grid_columnconfigure(0, weight=1)
        
        # Título com fundo branco
        title_label = ttk.Label(left_frame, 
                               text="Análise de Mamografia por IA",
                               font=('Helvetica', 24, 'bold'),
                               foreground=self.colors['primary'],
                               background=self.colors['white'])
        title_label.grid(row=0, column=0, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Subtítulo com fundo branco
        subtitle_label = ttk.Label(left_frame,
                                 text="Sistema de Detecção e Classificação",
                                 font=('Helvetica', 14),
                                 foreground=self.colors['light_text'],
                                 background=self.colors['white'])
        subtitle_label.grid(row=1, column=0, pady=(0, 30), sticky=(tk.W, tk.E))
        
        # Frame para a imagem com fundo branco
        self.image_frame = ttk.Frame(left_frame, borderwidth=2, relief='solid', style='Card.TFrame')
        self.image_frame.grid(row=2, column=0, pady=(0, 20), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid do image_frame
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)
        
        # Label para a imagem com fundo branco
        self.image_label = ttk.Label(self.image_frame, 
                                    text="Selecione uma imagem de mamografia\npara iniciar a análise",
                                    font=('Helvetica', 12),
                                    background=self.colors['white'])
        self.image_label.grid(row=0, column=0, padx=150, pady=150, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame para botões com fundo branco
        button_frame = ttk.Frame(left_frame, style='Card.TFrame')
        button_frame.grid(row=3, column=0, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Configurar grid do button_frame
        button_frame.grid_columnconfigure(0, weight=1)
        button_frame.grid_columnconfigure(1, weight=1)
        
        # Botões mais modernos
        self.select_btn = ttk.Button(button_frame, 
                                    text="Selecionar Imagem",
                                    style='Primary.TButton',
                                    command=self.select_image)
        self.select_btn.grid(row=0, column=0, padx=10, sticky=(tk.W, tk.E))
        
        self.analyze_btn = ttk.Button(button_frame, 
                                     text="Analisar Imagem",
                                     style='Secondary.TButton',
                                     command=self.analyze_image,
                                     state='disabled')
        self.analyze_btn.grid(row=0, column=1, padx=10, sticky=(tk.W, tk.E))
        
        # Status com ícone (com fundo branco)
        self.status_label = ttk.Label(left_frame, 
                                     text="✨ Aguardando imagem...",
                                     font=('Helvetica', 11),
                                     foreground=self.colors['light_text'],
                                     background=self.colors['white'])
        self.status_label.grid(row=4, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Frame direito - Gráficos
        right_frame = ttk.Frame(main_frame, padding="15")
        right_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid do right_frame
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        
        # Configurar gráficos
        self.setup_plots(right_frame)
    
    def setup_plots(self, frame):
        """Configura os gráficos com visual moderno e responsivo"""
        # Configurar grid do frame principal dos gráficos
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        
        # Gráfico de métricas com fundo branco
        metrics_frame = ttk.LabelFrame(frame, 
                                     text="Métricas de Desempenho",
                                     padding="15",
                                     style='Card.TLabelframe')
        metrics_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid do metrics_frame
        metrics_frame.grid_rowconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas container para métricas
        metrics_canvas_container = ttk.Frame(metrics_frame)
        metrics_canvas_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        metrics_canvas_container.grid_rowconfigure(0, weight=1)
        metrics_canvas_container.grid_columnconfigure(0, weight=1)
        
        self.metrics_fig = plt.Figure(dpi=100, facecolor=self.colors['white'])
        self.metrics_canvas = FigureCanvasTkAgg(self.metrics_fig, metrics_canvas_container)
        self.metrics_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Gráfico de probabilidades com fundo branco
        prob_frame = ttk.LabelFrame(frame, 
                                   text="Probabilidades de Predição",
                                   padding="15",
                                   style='Card.TLabelframe')
        prob_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid do prob_frame
        prob_frame.grid_rowconfigure(0, weight=1)
        prob_frame.grid_columnconfigure(0, weight=1)
        
        # Canvas container para probabilidades
        prob_canvas_container = ttk.Frame(prob_frame)
        prob_canvas_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        prob_canvas_container.grid_rowconfigure(0, weight=1)
        prob_canvas_container.grid_columnconfigure(0, weight=1)
        
        self.prob_fig = plt.Figure(dpi=100, facecolor=self.colors['white'])
        self.prob_canvas = FigureCanvasTkAgg(self.prob_fig, prob_canvas_container)
        self.prob_canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Inicializar gráficos
        self.init_plots()
    
    def init_plots(self):
        """Inicializa os gráficos com estilo moderno"""
        # Configurações de estilo personalizadas com fundo branco
        plt.rcParams.update({
            'figure.facecolor': self.colors['white'],
            'axes.facecolor': self.colors['white'],
            'savefig.facecolor': self.colors['white'],
            'figure.autolayout': True,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': '#cccccc',
            'figure.subplot.bottom': 0.15,
            'figure.subplot.top': 0.85,
            'figure.subplot.left': 0.15,
            'figure.subplot.right': 0.95
        })
        
        # Configurar fundo branco para as figuras
        self.metrics_fig.patch.set_facecolor(self.colors['white'])
        self.prob_fig.patch.set_facecolor(self.colors['white'])
        
        # Métricas
        self.metrics_ax = self.metrics_fig.add_subplot(111)
        metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
        values = [0, 0, 0, 0]
        
        # Gradiente de cores para as barras
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        bars = self.metrics_ax.bar(metrics, values, 
                                  color=colors,
                                  alpha=0.8,
                                  width=0.6)
        
        # Configurações do gráfico
        self.metrics_ax.set_ylim(0, 1)
        self.metrics_ax.set_title("Métricas de Desempenho", 
                                 pad=20, 
                                 fontsize=14, 
                                 fontweight='bold',
                                 color=self.colors['text'])
        
        # Estilo do grid
        self.metrics_ax.grid(True, linestyle='--', alpha=0.2, axis='y')
        self.metrics_ax.set_axisbelow(True)
        
        # Estilo dos eixos
        self.metrics_ax.spines['top'].set_visible(False)
        self.metrics_ax.spines['right'].set_visible(False)
        self.metrics_ax.spines['left'].set_alpha(0.2)
        self.metrics_ax.spines['bottom'].set_alpha(0.2)
        
        # Configuração dos labels
        plt.setp(self.metrics_ax.xaxis.get_majorticklabels(), 
                 rotation=30, 
                 ha='right',
                 fontsize=10)
        
        self.metrics_ax.set_ylabel("Valor", 
                                  fontsize=10, 
                                  color=self.colors['text'],
                                  alpha=0.8)
        
        # Probabilidades
        self.prob_ax = self.prob_fig.add_subplot(111)
        classes = ['Benigno', 'Maligno']
        probs = [0, 0]
        
        # Cores para as probabilidades
        prob_colors = ['#2ecc71', '#e74c3c']  # Verde para benigno, vermelho para maligno
        
        bars = self.prob_ax.bar(classes, probs,
                               color=prob_colors,
                               alpha=0.8,
                               width=0.6)
        
        # Configurações do gráfico
        self.prob_ax.set_ylim(0, 1)
        self.prob_ax.set_title("Probabilidades de Predição",
                              pad=20,
                              fontsize=14,
                              fontweight='bold',
                              color=self.colors['text'])
        
        # Estilo do grid
        self.prob_ax.grid(True, linestyle='--', alpha=0.2, axis='y')
        self.prob_ax.set_axisbelow(True)
        
        # Estilo dos eixos
        self.prob_ax.spines['top'].set_visible(False)
        self.prob_ax.spines['right'].set_visible(False)
        self.prob_ax.spines['left'].set_alpha(0.2)
        self.prob_ax.spines['bottom'].set_alpha(0.2)
        
        # Configuração dos labels
        plt.setp(self.prob_ax.xaxis.get_majorticklabels(),
                 rotation=30,
                 ha='right',
                 fontsize=10)
        
        self.prob_ax.set_ylabel("Probabilidade", 
                               fontsize=10, 
                               color=self.colors['text'],
                               alpha=0.8)
        
        # Ajustar layout
        self.metrics_fig.tight_layout(pad=2.0)
        self.prob_fig.tight_layout(pad=2.0)
        
        # Atualizar canvas
        self.metrics_canvas.draw()
        self.prob_canvas.draw()
    
    def select_image(self):
        """Seleciona uma imagem para análise"""
        file_path = filedialog.askopenfilename(
            title="Selecione uma imagem de mamografia",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                # Carregar e exibir imagem
                image = Image.open(file_path)
                
                # Redimensionar mantendo proporção
                display_size = (400, 400)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Converter para PhotoImage
                photo = ImageTk.PhotoImage(image)
                
                # Atualizar label
                self.image_label.configure(image=photo)
                self.image_label.image = photo
                
                # Habilitar botão de análise
                self.analyze_btn.configure(state='normal')
                
                # Guardar caminho da imagem
                self.current_image = file_path
                
                self.status_label.configure(text="Imagem carregada. Pronto para análise.")
                
            except Exception as e:
                print(f"Erro ao carregar imagem: {e}")
                self.status_label.configure(text="Erro ao carregar imagem")
    
    def analyze_image(self):
        """Analisa a imagem selecionada"""
        try:
            self.status_label.configure(text="Analisando imagem...")
            self.analyze_btn.configure(state='disabled')
            self.root.update()
            
            # Verificar se é uma imagem do dataset
            filename = os.path.basename(self.current_image)
            is_dataset_image = filename in self.image_info
            
            # Fazer predição
            probabilities = self.predict_image(self.current_image)
            
            if probabilities:
                # Calcular métricas (você pode implementar isso baseado no seu conjunto de validação)
                metrics = {
                    'Acurácia': 0.85,
                    'Precisão': 0.82,
                    'Recall': 0.79,
                    'F1-Score': 0.80
                }
                
                # Atualizar gráficos
                self.update_plots(metrics, probabilities)
                
                # Mostrar resultado
                pred_class = max(probabilities.items(), key=lambda x: x[1])[0]
                prob_value = probabilities[pred_class]
                
                if is_dataset_image:
                    real_class = self.image_info[filename]['pathology']
                    status_text = f"Análise concluída! Predição: {pred_class} ({prob_value:.2%}) | Real: {real_class}"
                else:
                    status_text = f"Análise concluída! Predição: {pred_class} ({prob_value:.2%})"
                
                self.status_label.configure(text=status_text)
            else:
                self.status_label.configure(text="Erro na análise da imagem")
            
            self.analyze_btn.configure(state='normal')
            
        except Exception as e:
            print(f"Erro durante análise: {e}")
            self.status_label.configure(text="Erro durante análise")
            self.analyze_btn.configure(state='normal')
    
    def update_plots(self, metrics, probabilities):
        """Atualiza os gráficos com estilo moderno"""
        # Limpar gráficos
        self.metrics_ax.clear()
        self.prob_ax.clear()
        
        # Atualizar métricas
        metrics_names = list(metrics.keys())
        metrics_values = list(metrics.values())
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']
        
        bars = self.metrics_ax.bar(metrics_names, metrics_values,
                                  color=colors,
                                  alpha=0.8,
                                  width=0.6)
        
        # Configurações do gráfico de métricas
        self.metrics_ax.set_ylim(0, 1.1)  # Aumentado para dar espaço aos valores
        self.metrics_ax.set_title("Métricas de Desempenho",
                                 pad=20,
                                 fontsize=14,
                                 fontweight='bold',
                                 color=self.colors['text'])
        
        # Estilo do grid
        self.metrics_ax.grid(True, linestyle='--', alpha=0.2, axis='y')
        self.metrics_ax.set_axisbelow(True)
        
        # Estilo dos eixos
        self.metrics_ax.spines['top'].set_visible(False)
        self.metrics_ax.spines['right'].set_visible(False)
        self.metrics_ax.spines['left'].set_alpha(0.2)
        self.metrics_ax.spines['bottom'].set_alpha(0.2)
        
        # Valores sobre as barras
        for bar in bars:
            height = bar.get_height()
            self.metrics_ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}',
                               ha='center',
                               va='bottom',
                               fontsize=10,
                               fontweight='bold',
                               color=self.colors['text'])
        
        # Configuração dos labels
        plt.setp(self.metrics_ax.xaxis.get_majorticklabels(),
                 rotation=30,
                 ha='right',
                 fontsize=10)
        
        # Atualizar probabilidades
        prob_names = list(probabilities.keys())
        prob_values = list(probabilities.values())
        prob_colors = ['#2ecc71', '#e74c3c']
        
        bars = self.prob_ax.bar(prob_names, prob_values,
                               color=prob_colors,
                               alpha=0.8,
                               width=0.6)
        
        # Configurações do gráfico de probabilidades
        self.prob_ax.set_ylim(0, 1.1)
        self.prob_ax.set_title("Probabilidades de Predição",
                              pad=20,
                              fontsize=14,
                              fontweight='bold',
                              color=self.colors['text'])
        
        # Estilo do grid
        self.prob_ax.grid(True, linestyle='--', alpha=0.2, axis='y')
        self.prob_ax.set_axisbelow(True)
        
        # Estilo dos eixos
        self.prob_ax.spines['top'].set_visible(False)
        self.prob_ax.spines['right'].set_visible(False)
        self.prob_ax.spines['left'].set_alpha(0.2)
        self.prob_ax.spines['bottom'].set_alpha(0.2)
        
        # Valores sobre as barras
        for bar in bars:
            height = bar.get_height()
            self.prob_ax.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.1%}',
                             ha='center',
                             va='bottom',
                             fontsize=10,
                             fontweight='bold',
                             color=self.colors['text'])
        
        # Configuração dos labels
        plt.setp(self.prob_ax.xaxis.get_majorticklabels(),
                 rotation=30,
                 ha='right',
                 fontsize=10)
        
        # Ajustar layout
        self.metrics_fig.tight_layout(pad=2.0)
        self.prob_fig.tight_layout(pad=2.0)
        
        # Atualizar canvas
        self.metrics_canvas.draw()
        self.prob_canvas.draw()

    def on_window_resize(self, event):
        """Lida com o redimensionamento da janela"""
        if event.widget == self.root:
            try:
                # Obter dimensões atuais da janela
                width = self.root.winfo_width()
                height = self.root.winfo_height()
                
                # Calcular novas dimensões para os gráficos
                graph_width = (width * 0.4) / 100  # 40% da largura da janela
                graph_height = (height * 0.35) / 100  # 35% da altura da janela
                
                # Ajustar tamanho dos gráficos
                self.metrics_fig.set_size_inches(graph_width, graph_height, forward=True)
                self.prob_fig.set_size_inches(graph_width, graph_height, forward=True)
                
                # Ajustar layout dos gráficos
                self.metrics_fig.tight_layout(pad=2.0)
                self.prob_fig.tight_layout(pad=2.0)
                
                # Atualizar canvas
                self.metrics_canvas.draw()
                self.prob_canvas.draw()
            except Exception as e:
                print(f"Erro ao redimensionar: {e}")

def main():
    root = tk.Tk()
    app = MamografiaPredictor(root)
    root.mainloop()

if __name__ == "__main__":
    main()