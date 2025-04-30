// Parámetros de la red
n_entradas = 13;  // Número de entradas en los datos médicos
n_ocultas = 10;   // Neuronas en la capa oculta
n_salidas = 4;    // Neuronas en la capa de salida (0: saludable, 1, 2, 3: enfermedad)
n_num_dat_ent = 30; // Número de datos de entrenamiento (ajustar según sea necesario)

//% Función de activación sigmoide
function y = sigmoid(x)
    y = 1 ./ (1 + exp(-x));
endfunction

//% Derivada de la sigmoide
function y = sigmoid_derivada(x)
    y = sigmoid(x) .* (1 - sigmoid(x));
endfunction

// Inicializar pesos y sesgos aleatoriamente
W1 = rand(n_entradas, n_ocultas);  // Pesos capa oculta
b1 = rand(1, n_ocultas);           // Sesgos capa oculta
W2 = rand(n_ocultas, n_salidas);   // Pesos capa salida
b2 = rand(1, n_salidas);           // Sesgos capa salida

// Cargar los datos desde archivo .data
file_path = "C:\Users\anali\Documents\6TO SEMESTRE\Reconocimiento de patrones\reprocessed.hungarian.data";
data = mopen(file_path, 'r'); // Abrir archivo en modo lectura

if data == -1 then
    disp("Error al abrir el archivo. Verifica la ruta.");
    quit; // Salir si no se puede abrir el archivo
else
    raw_data = mgetl(data); // Leer todas las líneas del archivo
    mclose(data); // Cerrar el archivo después de la lectura
end

// Procesar los datos
n_rows = size(raw_data, 'r'); // Número de filas
parsed_data = [];
for i = 1:n_rows
    line = strsubst(raw_data(i), ",", " "); // Reemplazar comas por espacios
    parsed_data = [parsed_data; evstr(line)]; // Convertir la línea en valores numéricos
end

// Manejar valores faltantes (-9)
parsed_data(parsed_data == -9) = 0; // Reemplazar -9 por 0 (o ajustarlo según convenga)

// Separar las características (X) y las etiquetas (Y)
X = parsed_data(:, 1:13); // 13 atributos (entrada)
Y_raw = parsed_data(:, 14); // Etiqueta de clase (salida)

// Codificar las etiquetas (Y) como valores one-hot (saludable: [1,0,0,0], etc.)
Y = zeros(size(Y_raw, 1), n_salidas); // Matriz de etiquetas
for i = 1:size(Y_raw, 1)
    if Y_raw(i) >= 0 & Y_raw(i) <= 3 then
        Y(i, Y_raw(i) + 1) = 1; // Convertir a representación one-hot
    end
end

// Mostrar datos de entrada
disp("Datos de entrada y etiquetas:");
disp(cat(2, X, Y));

// Entrenamiento
disp("Entrenamiento:");
//% Hiperparámetros
tasa_aprendizaje = 0.1;
max_iter = 1000;

//% Entrenamiento
for iter = 1:max_iter
    //% Propagación hacia adelante
    b1_expanded = repmat(b1, size(X, 1), 1);
    Z1 = X * W1 + b1_expanded; // Suma ponderada más sesgo
    A1 = sigmoid(Z1); // Salida de la capa oculta
    b2_expanded = repmat(b2, size(X, 1), 1);
    Z2 = A1 * W2 + b2_expanded; // Salida de la capa final
    A2 = sigmoid(Z2);

    // Cálculo del error
    error = Y - A2;

    // Retropropagación
    dZ2 = error .* sigmoid_derivada(Z2);
    dW2 = A1' * dZ2;
    db2 = sum(dZ2, 1);

    dZ1 = (dZ2 * W2') .* sigmoid_derivada(Z1);
    dW1 = X' * dZ1;
    db1 = sum(dZ1, 1);

    // Actualizar pesos y sesgos
    W2 = W2 + tasa_aprendizaje * dW2;
    b2 = b2 + tasa_aprendizaje * db2;
    W1 = W1 + tasa_aprendizaje * dW1;
    b1 = b1 + tasa_aprendizaje * db1;
end

// Probar la red
b1_exp = repmat(b1, size(X, 1), 1);
b2_exp = repmat(b2, size(X, 1), 1);
Y_pred = sigmoid(sigmoid(X * W1 + b1_exp) * W2 + b2_exp);

// Mostrar predicciones (redondeadas)
disp("Predicciones:");
disp(cat(2, X, fix(Y_pred + 0.5)));

