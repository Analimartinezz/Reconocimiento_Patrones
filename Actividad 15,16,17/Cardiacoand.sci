// Ruta del archivo de datos
file_path = "C:/Users/anali/Documents/6TO SEMESTRE/Reconocimiento de patrones/reprocessed.hungarian.data"; // Ajusta la ruta

// Abrir el archivo
data = mopen(file_path, 'r'); // Abrir el archivo en modo lectura

// Verificar si se abrió correctamente
if data == -1 then
    disp("Error al abrir el archivo. Verifica la ruta.");
    quit; // Salir si no se puede abrir el archivo
end

// Leer las líneas del archivo
raw_data = mgetl(data); // Leer todas las líneas

// Cerrar el archivo
mclose(data);

// Mostrar las primeras líneas de datos para verificar
disp("Primeras 5 líneas de datos:");
disp(raw_data(1:5)); // Muestra las primeras 5 líneas

// Procesar los datos
n_rows = size(raw_data, 'r'); // Número de filas
parsed_data = []; // Inicializar la matriz para los datos procesados

// Recorrer las filas para procesar
for i = 1:n_rows
    line = strsubst(raw_data(i), ",", " "); // Reemplazar las comas por espacios
    parsed_data = [parsed_data; evstr(line)]; // Convertir la línea en valores numéricos
end

// Mostrar las dimensiones de los datos procesados
disp("Dimensiones de los datos procesados:");
disp(size(parsed_data)); // Debe mostrar [filas, columnas]

// Manejar valores faltantes (-9)
parsed_data(parsed_data == -9) = 0; // Reemplazar -9 por 0 (o ajustarlo según convenga)

// Separar características (X) y etiquetas (Y)
X = parsed_data(:, 1:13); // Características (entradas)
Y_raw = parsed_data(:, 14); // Etiquetas de clase (salida)

// Convertir etiquetas (Y) a formato one-hot
Y = zeros(size(Y_raw, 1), 4); // Crear la matriz para etiquetas one-hot
for i = 1:size(Y_raw, 1)
    if Y_raw(i) >= 0 & Y_raw(i) <= 3 then
        Y(i, Y_raw(i) + 1) = 1; // Convertir a representación one-hot
    end
end

// Mostrar las primeras filas de X e Y para verificar
disp("Primeras 5 filas de X y Y:");
disp([X(1:5, :), Y(1:5, :)]);

// Parámetros de la red neuronal
N = [13, 4, 4]; // 13 entradas, 4 neuronas en la capa oculta, 4 salidas (one-hot)

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







