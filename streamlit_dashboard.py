import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="ML Analytics Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stMetric label {
        color: #1f1f1f !important;
        font-weight: 500 !important;
        opacity: 1 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #262730 !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #262730 !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #1f77b4 !important;
        font-weight: 500 !important;
    }
    /* Target all text elements in metric cards */
    .stMetric * {
        color: inherit;
    }
    .stMetric > div > div {
        color: #262730 !important;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 30px;
    }
    h2 {
        color: #2c3e50;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("<h1>🧬 Machine Learning Analytics Dashboard</h1>", unsafe_allow_html=True)
st.markdown("### Interactive Dimensionality Reduction & Clustering Analysis")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("📁 Upload CSV Dataset", type=['csv'])

if uploaded_file is not None:
    # Load data
    @st.cache_data
    def load_data(file):
        df = pd.read_csv(file)
        return df
    
    df = load_data(uploaded_file)
    
    st.sidebar.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Data preview
    with st.expander("📊 Dataset Preview", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**First 10 rows:**")
            st.dataframe(df.head(10), use_container_width=True)
        with col2:
            st.write("**Dataset Info:**")
            st.write(f"- Shape: {df.shape}")
            st.write(f"- Missing values: {df.isnull().sum().sum()}")
            st.write(f"- Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}")
            st.write(f"- Categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Target selection
    possible_targets = ['diagnosis', 'class', 'status', 'patient_status']
    available_targets = [col for col in possible_targets if col in df.columns]
    
    if available_targets:
        target_col = st.sidebar.selectbox("🎯 Select Target Column", available_targets)
    else:
        all_cols = df.columns.tolist()
        target_col = st.sidebar.selectbox("🎯 Select Target Column", all_cols)
    
    # Preprocessing
    @st.cache_data
    def preprocess_data(df, target_col):
        # Separate features and target
        X = df.drop(columns=[target_col, 'patient_id', 'date_of_surgery', 'date_of_last_visit', 'unnamed:_32'], errors='ignore')
        y = df[target_col]
        
        # Handle missing values in target and drop rows with NaN targets
        valid_mask = ~y.isna()
        X = X[valid_mask].copy()
        y = y[valid_mask].copy()
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        le = LabelEncoder()
        for col in categorical_cols:
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target
        le_target = LabelEncoder()
        if y.dtype == 'object':
            y_encoded = le_target.fit_transform(y.astype(str))
            class_names = le_target.classes_.tolist()
        else:
            y_encoded = y.values.copy()
            # Check for NaN in numeric target and handle
            if pd.isna(y_encoded).any():
                valid_mask = ~pd.isna(y_encoded)
                X = X[valid_mask].copy()
                y_encoded = y_encoded[valid_mask]
            # Convert to integer labels if numeric but represents classes
            unique_values = np.unique(y_encoded)
            if len(unique_values) <= 10:  # Likely a classification problem
                # Map to 0, 1, 2, ... labels
                mapping = {val: idx for idx, val in enumerate(unique_values)}
                y_encoded = np.array([mapping[val] for val in y_encoded])
                class_names = [f"Class {val}" for val in unique_values]
            else:
                class_names = [f"Class {i}" for i in unique_values]
        
        # Handle missing values in features - fill numeric columns with median, drop columns that are all NaN
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isna().all():
                X = X.drop(columns=[col])
            else:
                X[col] = X[col].fillna(X[col].median())
        
        # Fill remaining object columns with mode
        object_cols = X.select_dtypes(include=['object']).columns
        for col in object_cols:
            if X[col].isna().any():
                mode_value = X[col].mode()
                if len(mode_value) > 0:
                    X[col] = X[col].fillna(mode_value[0])
                else:
                    X[col] = X[col].fillna('unknown')
        
        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Final check - drop any remaining NaN rows
        if np.isnan(X_scaled).any():
            nan_mask = ~np.isnan(X_scaled).any(axis=1)
            X_scaled = X_scaled[nan_mask]
            y_encoded = y_encoded[nan_mask]
        
        # Ensure y_encoded is integer type for stratification
        y_encoded = y_encoded.astype(int)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        return X_train, X_test, y_train, y_test, X_scaled, y_encoded, class_names, X.columns.tolist()
    
    X_train, X_test, y_train, y_test, X_scaled, y_encoded, class_names, feature_names = preprocess_data(df, target_col)
    
    # Initialize model_results to avoid NameError
    if 'model_results' not in locals():
        model_results = {}
    
    # Sidebar - Dimensionality Reduction Settings
    st.sidebar.markdown("---")
    st.sidebar.header("🔬 Dimensionality Reduction")
    
    reduction_method = st.sidebar.selectbox(
        "Select Method",
        ["PCA", "LDA", "Autoencoder", "Compare All"]
    )
    
    if reduction_method == "PCA":
        variance_threshold = st.sidebar.slider("Variance to Retain", 0.8, 0.99, 0.95, 0.01)
        n_components_viz = st.sidebar.slider("Components for Visualization", 2, 3, 2)
    elif reduction_method == "LDA":
        n_classes = len(np.unique(y_encoded))
        max_components = min(n_classes - 1, X_train.shape[1])
        if max_components <= 2:
            n_components_viz = max_components if max_components > 0 else 1
        else:
            n_components_viz = st.sidebar.slider("Components for Visualization", 2, min(3, max_components), 2)
    elif reduction_method == "Autoencoder":
        encoding_dim = st.sidebar.slider("Encoding Dimension", 5, 50, 20, 5)
        epochs = st.sidebar.slider("Training Epochs", 10, 100, 50, 10)
    
    # Sidebar - Clustering Settings
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Clustering")
    
    clustering_method = st.sidebar.selectbox(
        "Select Clustering Algorithm",
        ["KMeans", "DBSCAN", "Agglomerative", "None"]
    )
    
    if clustering_method == "KMeans":
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    elif clustering_method == "DBSCAN":
        eps = st.sidebar.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1)
        min_samples = st.sidebar.slider("Min Samples", 2, 20, 5)
    elif clustering_method == "Agglomerative":
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
        linkage = st.sidebar.selectbox("Linkage", ["ward", "complete", "average"])
    
    # Model Selection
    st.sidebar.markdown("---")
    st.sidebar.header("🤖 Model Training")
    model_options = st.sidebar.multiselect(
        "Select Models to Train",
        ["Logistic Regression", "SVM", "KNN", "Decision Tree"],
        default=["Logistic Regression", "SVM"]
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dimensionality Reduction",
        "🎯 Clustering Analysis",
        "🤖 Model Performance",
        "📈 Comparison Dashboard",
        "📋 Data Insights"
    ])
    
    # ========================================================================
    # TAB 1: DIMENSIONALITY REDUCTION
    # ========================================================================
    with tab1:
        st.header("📊 Dimensionality Reduction Analysis")
        
        if reduction_method in ["PCA", "Compare All"]:
            st.subheader("🔵 Principal Component Analysis (PCA)")
            
            # Perform PCA
            pca_full = PCA()
            X_train_pca_full = pca_full.fit_transform(X_train)
            
            pca = PCA(n_components=variance_threshold if reduction_method == "PCA" else 0.95)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Features", X_train.shape[1])
            with col2:
                st.metric("PCA Components", X_train_pca.shape[1])
            with col3:
                st.metric("Variance Retained", f"{pca.explained_variance_ratio_.sum():.2%}")
            with col4:
                reduction = (1 - X_train_pca.shape[1]/X_train.shape[1]) * 100
                st.metric("Dimension Reduction", f"{reduction:.1f}%")
            
            # Scree plot
            col1, col2 = st.columns(2)
            
            with col1:
                explained_var = pca_full.explained_variance_ratio_[:20]
                fig_scree = go.Figure()
                fig_scree.add_trace(go.Bar(
                    x=list(range(1, len(explained_var)+1)),
                    y=explained_var,
                    marker_color='steelblue',
                    name='Explained Variance'
                ))
                fig_scree.update_layout(
                    title="PCA Scree Plot",
                    xaxis_title="Principal Component",
                    yaxis_title="Explained Variance Ratio",
                    height=400
                )
                st.plotly_chart(fig_scree, use_container_width=True)
            
            with col2:
                cumulative_var = np.cumsum(pca_full.explained_variance_ratio_[:20])
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=list(range(1, len(cumulative_var)+1)),
                    y=cumulative_var,
                    mode='lines+markers',
                    marker=dict(size=8, color='coral'),
                    line=dict(width=3),
                    name='Cumulative Variance'
                ))
                fig_cum.add_hline(y=0.95, line_dash="dash", line_color="green", 
                                 annotation_text="95% Threshold")
                fig_cum.update_layout(
                    title="Cumulative Explained Variance",
                    xaxis_title="Number of Components",
                    yaxis_title="Cumulative Variance",
                    height=400
                )
                st.plotly_chart(fig_cum, use_container_width=True)
            
            # 2D/3D Visualization
            if reduction_method == "PCA":
                pca_viz = PCA(n_components=n_components_viz)
            else:
                pca_viz = PCA(n_components=2)
            
            X_train_pca_viz = pca_viz.fit_transform(X_train)
            
            if pca_viz.n_components_ == 2:
                df_pca = pd.DataFrame(
                    X_train_pca_viz,
                    columns=['PC1', 'PC2']
                )
                df_pca['Class'] = [class_names[int(i)] for i in y_train]
                
                fig_2d = px.scatter(
                    df_pca, x='PC1', y='PC2', color='Class',
                    title='PCA 2D Projection',
                    labels={'PC1': f'PC1 ({pca_viz.explained_variance_ratio_[0]:.2%})',
                           'PC2': f'PC2 ({pca_viz.explained_variance_ratio_[1]:.2%})'},
                    height=500
                )
                st.plotly_chart(fig_2d, use_container_width=True)
            
            elif pca_viz.n_components_ == 3:
                df_pca = pd.DataFrame(
                    X_train_pca_viz,
                    columns=['PC1', 'PC2', 'PC3']
                )
                df_pca['Class'] = [class_names[int(i)] for i in y_train]
                
                fig_3d = px.scatter_3d(
                    df_pca, x='PC1', y='PC2', z='PC3', color='Class',
                    title='PCA 3D Projection',
                    labels={'PC1': f'PC1 ({pca_viz.explained_variance_ratio_[0]:.2%})',
                           'PC2': f'PC2 ({pca_viz.explained_variance_ratio_[1]:.2%})',
                           'PC3': f'PC3 ({pca_viz.explained_variance_ratio_[2]:.2%})'},
                    height=600
                )
                st.plotly_chart(fig_3d, use_container_width=True)
        
        if reduction_method in ["LDA", "Compare All"]:
            n_classes = len(np.unique(y_encoded))
            # LDA works for both binary (2 classes) and multiclass (>2 classes)
            if n_classes >= 2:
                st.markdown("---")
                st.subheader("🟢 Linear Discriminant Analysis (LDA)")
                
                # Perform LDA
                max_components = min(n_classes - 1, X_train.shape[1])
                lda = LDA(n_components=max_components)
                X_train_lda = lda.fit_transform(X_train, y_train)
                X_test_lda = lda.transform(X_test)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("LDA Components", X_train_lda.shape[1])
                with col2:
                    variance_explained = lda.explained_variance_ratio_.sum() if hasattr(lda, 'explained_variance_ratio_') else 1.0
                    st.metric("Variance Explained", f"{variance_explained:.2%}")
                with col3:
                    st.metric("Classes", n_classes)
                
                # LDA visualization
                if X_train_lda.shape[1] >= 2:
                    df_lda = pd.DataFrame(
                        X_train_lda[:, :2],
                        columns=['LD1', 'LD2']
                    )
                    df_lda['Class'] = [class_names[int(i)] for i in y_train]
                    
                    fig_lda = px.scatter(
                        df_lda, x='LD1', y='LD2', color='Class',
                        title='LDA 2D Projection (Supervised)',
                        labels={'LD1': f'LD1 ({lda.explained_variance_ratio_[0]:.2%})' if len(lda.explained_variance_ratio_) > 0 else 'LD1',
                               'LD2': f'LD2 ({lda.explained_variance_ratio_[1]:.2%})' if len(lda.explained_variance_ratio_) > 1 else 'LD2'},
                        height=500
                    )
                    st.plotly_chart(fig_lda, use_container_width=True)
                elif X_train_lda.shape[1] == 1:
                    # For binary classification, show 1D LDA projection as histogram
                    df_lda = pd.DataFrame(
                        X_train_lda,
                        columns=['LD1']
                    )
                    df_lda['Class'] = [class_names[int(i)] for i in y_train]
                    
                    # Create histogram/bar chart for 1D LDA
                    fig_lda = px.histogram(
                        df_lda, x='LD1', color='Class',
                        title='LDA 1D Projection (Binary Classification)',
                        labels={'LD1': 'Linear Discriminant 1'},
                        height=500,
                        nbins=30,
                        barmode='overlay',
                        opacity=0.7
                    )
                    st.plotly_chart(fig_lda, use_container_width=True)
                    
                    # Also show box plot for better visualization
                    fig_box = px.box(
                        df_lda, x='Class', y='LD1',
                        title='LDA Distribution by Class',
                        height=400
                    )
                    st.plotly_chart(fig_box, use_container_width=True)
        
        if reduction_method in ["Autoencoder", "Compare All"]:
            st.markdown("---")
            st.subheader("🟣 Autoencoder (Deep Learning)")
            
            # Build autoencoder
            if reduction_method == "Autoencoder":
                enc_dim = encoding_dim
                n_epochs = epochs
            else:
                enc_dim = 20
                n_epochs = 30
            
            with st.spinner("Training autoencoder..."):
                input_layer = Input(shape=(X_train.shape[1],))
                encoded = Dense(128, activation='relu')(input_layer)
                encoded = Dense(64, activation='relu')(encoded)
                encoded = Dense(enc_dim, activation='relu', name='encoding')(encoded)
                decoded = Dense(64, activation='relu')(encoded)
                decoded = Dense(128, activation='relu')(decoded)
                decoded = Dense(X_train.shape[1], activation='linear')(decoded)
                
                autoencoder = Model(input_layer, decoded)
                encoder = Model(input_layer, autoencoder.get_layer('encoding').output)
                
                autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
                
                early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                history = autoencoder.fit(
                    X_train, X_train,
                    epochs=n_epochs,
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=0
                )
                
                X_train_ae = encoder.predict(X_train, verbose=0)
                X_test_ae = encoder.predict(X_test, verbose=0)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Encoding Dimension", enc_dim)
            with col2:
                st.metric("Training Epochs", len(history.history['loss']))
            with col3:
                st.metric("Final Loss", f"{history.history['loss'][-1]:.6f}")
            
            # Training history
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(
                y=history.history['loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue', width=2)
            ))
            fig_history.add_trace(go.Scatter(
                y=history.history['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red', width=2)
            ))
            fig_history.update_layout(
                title="Autoencoder Training History",
                xaxis_title="Epoch",
                yaxis_title="MSE Loss",
                height=400
            )
            st.plotly_chart(fig_history, use_container_width=True)
            
            # 2D visualization
            ae_pca = PCA(n_components=2)
            X_train_ae_2d = ae_pca.fit_transform(X_train_ae)
            
            df_ae = pd.DataFrame(
                X_train_ae_2d,
                columns=['AE1', 'AE2']
            )
            df_ae['Class'] = [class_names[int(i)] for i in y_train]
            
            fig_ae = px.scatter(
                df_ae, x='AE1', y='AE2', color='Class',
                title='Autoencoder Features - 2D Projection',
                height=500
            )
            st.plotly_chart(fig_ae, use_container_width=True)
    
    # ========================================================================
    # TAB 2: CLUSTERING ANALYSIS
    # ========================================================================
    with tab2:
        st.header("🎯 Clustering Analysis")
        
        if clustering_method != "None":
            # Calculate n_classes for LDA
            n_classes = len(np.unique(y_encoded))
            
            # Select data for clustering
            cluster_data_source = st.selectbox(
                "Select Data for Clustering",
                ["Original Features", "PCA Features", "LDA Features", "Autoencoder Features"]
            )
            
            if cluster_data_source == "Original Features":
                X_cluster = X_scaled
            elif cluster_data_source == "PCA Features":
                pca_cluster = PCA(n_components=0.95)
                X_cluster = pca_cluster.fit_transform(X_scaled)
            elif cluster_data_source == "LDA Features" and n_classes >= 2:
                lda_cluster = LDA()
                X_cluster = lda_cluster.fit_transform(X_scaled, y_encoded)
            else:
                # Use autoencoder features
                X_cluster = X_train_ae if 'X_train_ae' in locals() else X_scaled
            
            # Perform clustering
            if clustering_method == "KMeans":
                clusterer = KMeans(n_clusters=n_clusters, random_state=42)
                cluster_labels = clusterer.fit_predict(X_cluster)
                
            elif clustering_method == "DBSCAN":
                clusterer = DBSCAN(eps=eps, min_samples=min_samples)
                cluster_labels = clusterer.fit_predict(X_cluster)
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                
            elif clustering_method == "Agglomerative":
                clusterer = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                cluster_labels = clusterer.fit_predict(X_cluster)
            
            # Metrics
            if len(set(cluster_labels)) > 1:
                silhouette = silhouette_score(X_cluster, cluster_labels)
                davies_bouldin = davies_bouldin_score(X_cluster, cluster_labels)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Clusters", len(set(cluster_labels)))
                with col2:
                    st.metric("Silhouette Score", f"{silhouette:.3f}")
                with col3:
                    st.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}")
                
                # Visualization - handle cases with fewer dimensions
                n_features_cluster = X_cluster.shape[1]
                
                if n_features_cluster >= 2:
                    # Use PCA to reduce to 2D for visualization
                    n_components_viz = min(2, n_features_cluster)
                    pca_2d_cluster = PCA(n_components=n_components_viz)
                    X_cluster_2d = pca_2d_cluster.fit_transform(X_cluster)
                    
                    df_cluster = pd.DataFrame(
                        X_cluster_2d,
                        columns=[f'Component {i+1}' for i in range(n_components_viz)]
                    )
                    df_cluster['Cluster'] = cluster_labels.astype(str)
                    df_cluster['True Class'] = [class_names[int(i)] for i in y_encoded[:len(cluster_labels)]]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if n_components_viz == 2:
                            fig_cluster = px.scatter(
                                df_cluster, x='Component 1', y='Component 2', 
                                color='Cluster',
                                title=f'{clustering_method} Clustering Results',
                                height=500
                            )
                        else:
                            fig_cluster = px.scatter(
                                df_cluster, x='Component 1', 
                                color='Cluster',
                                title=f'{clustering_method} Clustering Results (1D)',
                                height=500
                            )
                        st.plotly_chart(fig_cluster, use_container_width=True)
                    
                    with col2:
                        if n_components_viz == 2:
                            fig_true = px.scatter(
                                df_cluster, x='Component 1', y='Component 2', 
                                color='True Class',
                                title='True Class Labels',
                                height=500
                            )
                        else:
                            fig_true = px.scatter(
                                df_cluster, x='Component 1', 
                                color='True Class',
                                title='True Class Labels (1D)',
                                height=500
                            )
                        st.plotly_chart(fig_true, use_container_width=True)
                else:
                    # Handle 1D case with histogram/box plot
                    df_cluster = pd.DataFrame(
                        X_cluster.reshape(-1, 1),
                        columns=['Component 1']
                    )
                    df_cluster['Cluster'] = cluster_labels.astype(str)
                    df_cluster['True Class'] = [class_names[int(i)] for i in y_encoded[:len(cluster_labels)]]
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_cluster = px.histogram(
                            df_cluster, x='Component 1', color='Cluster',
                            title=f'{clustering_method} Clustering Results (1D)',
                            height=500,
                            nbins=30,
                            barmode='overlay',
                            opacity=0.7
                        )
                        st.plotly_chart(fig_cluster, use_container_width=True)
                    
                    with col2:
                        fig_true = px.histogram(
                            df_cluster, x='Component 1', color='True Class',
                            title='True Class Labels (1D)',
                            height=500,
                            nbins=30,
                            barmode='overlay',
                            opacity=0.7
                        )
                        st.plotly_chart(fig_true, use_container_width=True)
                
                # Cluster distribution
                cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
                fig_dist = px.bar(
                    x=cluster_counts.index.astype(str),
                    y=cluster_counts.values,
                    labels={'x': 'Cluster', 'y': 'Count'},
                    title='Cluster Size Distribution',
                    color=cluster_counts.values,
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("👆 Select a clustering algorithm from the sidebar to begin analysis")
    
    # ========================================================================
    # TAB 3: MODEL PERFORMANCE
    # ========================================================================
    with tab3:
        st.header("🤖 Model Performance Evaluation")
        
        if len(model_options) > 0:
            # Store results
            model_results = {}
            
            # Define models
            models_dict = {
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "SVM": SVC(kernel='rbf', random_state=42, probability=True),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42)
            }
            
            # Train models on different feature sets
            feature_sets = {
                "Original": (X_train, X_test),
                "PCA": (X_train_pca, X_test_pca) if 'X_train_pca' in locals() else None,
                "LDA": (X_train_lda, X_test_lda) if 'X_train_lda' in locals() else None,
                "Autoencoder": (X_train_ae, X_test_ae) if 'X_train_ae' in locals() else None
            }
            
            with st.spinner("Training models..."):
                for model_name in model_options:
                    model_results[model_name] = {}
                    
                    for fs_name, fs_data in feature_sets.items():
                        if fs_data is not None:
                            X_tr, X_te = fs_data
                            model = models_dict[model_name]
                            model.fit(X_tr, y_train)
                            y_pred = model.predict(X_te)
                            
                            accuracy = accuracy_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred, average='weighted')
                            
                            model_results[model_name][fs_name] = {
                                'accuracy': accuracy,
                                'f1': f1,
                                'y_pred': y_pred
                            }
            
            # Display results
            st.subheader("📊 Performance Metrics")
            
            # Create comparison dataframe
            comparison_data = []
            for model_name, fs_results in model_results.items():
                for fs_name, metrics in fs_results.items():
                    comparison_data.append({
                        'Model': model_name,
                        'Feature Set': fs_name,
                        'Accuracy': metrics['accuracy'],
                        'F1 Score': metrics['f1']
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Metrics visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig_acc = px.bar(
                    comparison_df, x='Model', y='Accuracy', color='Feature Set',
                    barmode='group',
                    title='Model Accuracy Comparison',
                    height=400
                )
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                fig_f1 = px.bar(
                    comparison_df, x='Model', y='F1 Score', color='Feature Set',
                    barmode='group',
                    title='Model F1 Score Comparison',
                    height=400
                )
                st.plotly_chart(fig_f1, use_container_width=True)
            
            # Heatmap
            pivot_acc = comparison_df.pivot(index='Model', columns='Feature Set', values='Accuracy')
            fig_heatmap = px.imshow(
                pivot_acc,
                labels=dict(x="Feature Set", y="Model", color="Accuracy"),
                title="Performance Heatmap - Accuracy",
                color_continuous_scale='RdYlGn',
                aspect='auto',
                height=400
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Best model
            best_result = comparison_df.loc[comparison_df['F1 Score'].idxmax()]
            st.success(f"🏆 **Best Model:** {best_result['Model']} with {best_result['Feature Set']} features (F1: {best_result['F1 Score']:.4f})")
            
            # Confusion Matrix
            st.subheader("📊 Confusion Matrix - Best Model")
            best_model_name = best_result['Model']
            best_fs_name = best_result['Feature Set']
            y_pred_best = model_results[best_model_name][best_fs_name]['y_pred']
            
            cm = confusion_matrix(y_test, y_pred_best)
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=class_names,
                y=class_names,
                title=f"Confusion Matrix - {best_model_name} ({best_fs_name})",
                color_continuous_scale='Blues',
                text_auto=True,
                height=500
            )
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("👆 Select models from the sidebar to train and evaluate")
    
    # ========================================================================
    # TAB 4: COMPARISON DASHBOARD
    # ========================================================================
    with tab4:
        st.header("📈 Comprehensive Comparison Dashboard")
        
        # Summary metrics
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Features",
                X_train.shape[1],
                delta=None
            )
        
        with col2:
            if 'X_train_pca' in locals():
                reduction = ((X_train.shape[1] - X_train_pca.shape[1]) / X_train.shape[1]) * 100
                st.metric(
                    "PCA Reduction",
                    f"{X_train_pca.shape[1]} dims",
                    delta=f"-{reduction:.1f}%"
                )
        
        with col3:
            if len(model_results) > 0:
                best_f1 = comparison_df['F1 Score'].max()
                st.metric(
                    "Best F1 Score",
                    f"{best_f1:.4f}",
                    delta=None
                )
        
        with col4:
            st.metric(
                "Classes",
                len(class_names),
                delta=None
            )
        
        # Feature reduction comparison
        if 'X_train_pca' in locals() and 'X_train_lda' in locals() and 'X_train_ae' in locals():
            st.subheader("📊 Dimensionality Reduction Comparison")
            
            reduction_comparison = pd.DataFrame({
                'Method': ['Original', 'PCA', 'LDA', 'Autoencoder'],
                'Dimensions': [X_train.shape[1], X_train_pca.shape[1], X_train_lda.shape[1], X_train_ae.shape[1]],
                'Reduction (%)': [
                    0,
                    ((X_train.shape[1] - X_train_pca.shape[1]) / X_train.shape[1]) * 100,
                    ((X_train.shape[1] - X_train_lda.shape[1]) / X_train.shape[1]) * 100,
                    ((X_train.shape[1] - X_train_ae.shape[1]) / X_train.shape[1]) * 100
                ]
            })
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dims = px.bar(
                    reduction_comparison,
                    x='Method',
                    y='Dimensions',
                    title='Number of Dimensions by Method',
                    color='Dimensions',
                    color_continuous_scale='viridis',
                    text='Dimensions',
                    height=400
                )
                fig_dims.update_traces(textposition='outside')
                st.plotly_chart(fig_dims, use_container_width=True)
            
            with col2:
                fig_reduction = px.bar(
                    reduction_comparison[reduction_comparison['Method'] != 'Original'],
                    x='Method',
                    y='Reduction (%)',
                    title='Dimensionality Reduction Percentage',
                    color='Reduction (%)',
                    color_continuous_scale='Reds',
                    text='Reduction (%)',
                    height=400
                )
                fig_reduction.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                st.plotly_chart(fig_reduction, use_container_width=True)
        
        # Model performance overview
        if len(model_results) > 0:
            st.subheader("🤖 Model Performance Overview")
            
            # Create radar chart
            fig_radar = go.Figure()
            
            for model_name in model_options:
                if model_name in model_results:
                    fs_results = model_results[model_name]
                    metrics = [fs_results.get(fs, {}).get('f1', 0) for fs in ['Original', 'PCA', 'LDA', 'Autoencoder']]
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=metrics,
                        theta=['Original', 'PCA', 'LDA', 'Autoencoder'],
                        fill='toself',
                        name=model_name
                    ))
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title="Model Performance Across Feature Sets (F1 Score)",
                height=500
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Performance table
            st.dataframe(
                comparison_df.style.highlight_max(subset=['Accuracy', 'F1 Score'], axis=0),
                use_container_width=True
            )
    
    # ========================================================================
    # TAB 5: DATA INSIGHTS
    # ========================================================================
    with tab5:
        st.header("📋 Data Insights & Statistics")
        
        # Dataset statistics
        st.subheader("📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Basic Statistics:**")
            stats_df = pd.DataFrame({
                'Metric': ['Total Samples', 'Training Samples', 'Test Samples', 'Features', 'Classes'],
                'Value': [len(df), X_train.shape[0], X_test.shape[0], X_train.shape[1], len(class_names)]
            })
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.write("**Class Distribution:**")
            class_dist = pd.Series(y_encoded).value_counts().sort_index()
            class_dist_df = pd.DataFrame({
                'Class': [class_names[i] for i in class_dist.index],
                'Count': class_dist.values,
                'Percentage': (class_dist.values / len(y_encoded) * 100).round(2)
            })
            st.dataframe(class_dist_df, use_container_width=True)
        
        # Class distribution visualization
        fig_class_dist = px.pie(
            values=class_dist.values,
            names=[class_names[i] for i in class_dist.index],
            title='Class Distribution',
            height=400
        )
        st.plotly_chart(fig_class_dist, use_container_width=True)
        
        # Feature correlation (top features)
        st.subheader("🔗 Feature Correlation Analysis")
        
        # Select top 20 features for correlation
        X_df = pd.DataFrame(X_scaled[:500], columns=feature_names)
        corr_matrix = X_df.iloc[:, :min(20, len(feature_names))].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            title="Feature Correlation Heatmap (Top 20 Features)",
            color_continuous_scale='RdBu_r',
            aspect='auto',
            height=600
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Feature importance (if available)
        if 'X_train_pca' in locals():
            st.subheader("📈 Feature Importance (PCA)")
            
            # Get top features from first PC
            pc1_weights = np.abs(pca.components_[0])
            top_indices = np.argsort(pc1_weights)[-15:][::-1]
            
            importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_indices],
                'Importance': pc1_weights[top_indices]
            })
            
            fig_importance = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Top 15 Features Contributing to PC1',
                color='Importance',
                color_continuous_scale='viridis',
                height=500
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Download results
        st.subheader("💾 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if len(model_results) > 0:
                csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Model Results",
                    data=csv,
                    file_name="model_performance.csv",
                    mime="text/csv"
                )
        
        with col2:
            if 'X_train_pca' in locals():
                pca_results = pd.DataFrame(X_train_pca, columns=[f'PC{i+1}' for i in range(X_train_pca.shape[1])])
                pca_csv = pca_results.to_csv(index=False)
                st.download_button(
                    label="📥 Download PCA Features",
                    data=pca_csv,
                    file_name="pca_features.csv",
                    mime="text/csv"
                )
        
        with col3:
            try:
                if len(model_results) > 0:
                    best_model_name = best_result['Model']
                    best_fs_name = best_result['Feature Set']
                    best_f1_str = f"{best_result['F1 Score']:.4f}"
                else:
                    best_model_name = 'N/A'
                    best_fs_name = 'N/A'
                    best_f1_str = 'N/A'
            except (NameError, KeyError):
                best_model_name = 'N/A'
                best_fs_name = 'N/A'
                best_f1_str = 'N/A'
            
            summary = f"""
            Dataset Summary
            ===============
            Total Samples: {len(df)}
            Features: {X_train.shape[1]}
            Classes: {len(class_names)}
            
            Best Model: {best_model_name}
            Best Feature Set: {best_fs_name}
            Best F1 Score: {best_f1_str}
            """
            st.download_button(
                label="📥 Download Summary",
                data=summary,
                file_name="analysis_summary.txt",
                mime="text/plain"
            )

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to the ML Analytics Dashboard!
    
    This interactive dashboard allows you to:
    
    ### 🔬 Dimensionality Reduction
    - **PCA (Principal Component Analysis)** - Unsupervised linear reduction
    - **LDA (Linear Discriminant Analysis)** - Supervised class separation
    - **Autoencoder** - Deep learning non-linear reduction
    
    ### 🎯 Clustering Analysis
    - **KMeans** - Partition-based clustering
    - **DBSCAN** - Density-based clustering
    - **Agglomerative** - Hierarchical clustering
    
    ### 🤖 Model Training
    - Train multiple models on different feature sets
    - Compare performance across dimensionality reduction methods
    - Visualize confusion matrices and metrics
    
    ### 📊 Interactive Visualizations
    - 2D/3D scatter plots
    - Performance comparisons
    - Feature importance analysis
    - Real-time clustering
    
    ---
    
    ### 🚀 Getting Started
    
    1. **Upload your CSV dataset** using the sidebar
    2. **Configure** dimensionality reduction parameters
    3. **Select** clustering and model options
    4. **Explore** the interactive visualizations
    5. **Download** results and insights
    
    ---
    
    ### 📋 Dataset Requirements
    
    Your CSV should contain:
    - Numeric features for analysis
    - A target column (diagnosis, class, status, etc.)
    - Optional: patient_id, dates, categorical features
    
    ---
    
    ### 💡 Tips
    
    - Start with PCA to understand variance structure
    - Use LDA for supervised classification tasks
    - Try Autoencoder for complex non-linear patterns
    - Compare all methods in the Comparison Dashboard
    - Export results for further analysis
    
    ---
    
    **👈 Upload your dataset in the sidebar to begin!**
    """)
    
    # Example dataset info
    with st.expander("📖 Example: Expected Dataset Format"):
        example_df = pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003'],
            'age': [45, 52, 38],
            'protein1': [0.85, 0.92, 0.78],
            'protein2': [1.23, 1.45, 1.12],
            'diagnosis': ['Malignant', 'Benign', 'Malignant']
        })
        st.dataframe(example_df, use_container_width=True)
        st.caption("Your dataset should have numeric features and a target column")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🧬 ML Analytics Dashboard | Built with Streamlit & Plotly</p>
        <p>💡 For research, education, and data science projects</p>
    </div>
    """, unsafe_allow_html=True)