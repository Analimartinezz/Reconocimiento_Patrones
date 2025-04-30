// Red neuronal para clasificación IRIS con 4 entradas, 10 neuronas ocultas y 3 salidas

// Definir la arquitectura de la red
N = [4, 10, 3]; // 4 entradas, 10 neuronas ocultas, 3 salidas

// Cargar los datos desde el archivo .sce que contiene IRIS_DATA
exec("C:\Users\anali\Documents\6TO SEMESTRE\Reconocimiento de patrones\iris.sce", -1);

// Separar características (X) y etiquetas (Y)
X = IRIS_DATA(:, 1:4);
n_num_dat_ent = size(X, 1); // número de muestras, debería ser 150

// Crear etiquetas codificadas en caliente (one-hot encoding)
Y = zeros(n_num_dat_ent, 3);
for i = 1:n_num_dat_ent
    clase = IRIS_DATA(i, 5); // 0, 1 o 2
    Y(i, clase + 1) = 1;     // convierte a [1 0 0]  Setosa, [0 1 0] versicolor, o [0 0 1] virginica
end

// Mostrar tamaños para verificación
disp("Tamaño de X: "), disp(size(X));
disp("Tamaño de Y: "), disp(size(Y));

// Parámetros de entrenamiento
learning_rate = [0.1, 0]; // tasa de aprendizaje
epochs = 1000;            // número de épocas

// Inicializar pesos y sesgos aleatoriamente
W = ann_FF_init(N); // Inicializa la red

// Parámetros de entrenamiento
learning_rate = [0.1, 0]; // Tasa de aprendizaje
epochs = 100; // Número de épocas de entrenamiento

// Entrenamiento de la red utilizando aprendizaje en línea
W = ann_FF_Std_online(X', Y', N, W, learning_rate, epochs); // Entrenar la red

// Probar la red
predicciones = ann_FF_run(X', N, W); // Realizar predicciones

// Mostrar las predicciones
disp("Predicciones de la red neuronal:");
disp(predicciones);

// Convertir las salidas de la red a 0 o 1
AUX = sign(predicciones' - 0.5); // Redondear las salidas
P = (AUX + 1) / 2; // Convertir de -1, 1 a 0, 1

// Mostrar las predicciones finales
disp("Predicciones finales:");
disp([X, P]);

