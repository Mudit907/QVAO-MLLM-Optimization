# Uninstall conflicting packages to avoid issues
!pip uninstall -y pathos dopamine-rl sentence-transformers plotnine mlxtend cesium bigframes gcsfs -q

# Install dependencies with resolved versions
!pip install -q transformers==4.41.1 accelerate==0.30.0 bitsandbytes==0.43.1 sentencepiece==0.2.0 peft==0.11.1 evaluate==0.4.2 datasets==2.14.0 qiskit==1.0.2 pillow==10.3.0 opencv-python==4.10.0.84 numpy==1.26.4 rich==13.7.1 fsspec==2025.3.2 dill==0.3.8 multiprocess==0.70.16 gymnasium==1.0.0 matplotlib==3.8.4 scikit-learn==1.3.2

# Verify installations
try:
    import transformers, accelerate, bitsandbytes, peft, datasets, qiskit, PIL, cv2, numpy, rich, fsspec, dill, multiprocess, gymnasium, matplotlib, sklearn
    print("Transformers:", transformers.__version__)
    print("Datasets:", datasets.__version__)
    print("NumPy:", numpy.__version__)
    print("Pillow:", PIL.__version__)
    print("Rich:", rich.__version__)
    print("FSSpec:", fsspec.__version__)
    print("Dill:", dill.__version__)
    print("Multiprocess:", multiprocess.__version__)
    print("Gymnasium:", gymnasium.__version__)
    print("Matplotlib:", matplotlib.__version__)
    print("Scikit-learn:", sklearn.__version__)
    print("All modules loaded successfully!")
except Exception as e:
    print(f"Error loading modules: {e}")
