# IRESNs

Questa libreria permette di costruire modelli di tipo ESN usando tensorflow.

Modelli supportati:

 - ESN (Echo State Network)
 - IRESN (Input Routed ESN)
 - IIRESN (Interconnected Input Routed ESN)
 - IIRESNvsr (Interconnected Input Routed ESN variable size reservoirs)
 - e altre combinazioni.

Codice di esempio con spiegazione in `main.ipynb`

Libreria utilizzata in questo progetto: https://github.com/SilverLuke/Tesi

# Funzionamento

Come visualizzare la creazione del reservoir, per i modelli IRESN, IIRESN, IIRESNvsr:

![awesome diagram](diagram.png)

Il numero di righe del kernel dipende dalle `N` feature di cui è composto il dataset, 
lo stesso vale per il numero di divisioni verticali. I valori x, y e z fanno riferimento a quanti elementi ha il vettore, 
e vale sia per il kernel che per il recurrent kernel.
La somma dei valori x + y + z = numero di unità fornito quando si costruisce il modello.

Nel kernel le sotto matrici con la riga nera sono settate a 0, mentre le sotto matrici rosse vengono istanziate
tramite `tf.keras.initalizers.RandomUniform()`.

Il recurrent kernel viene diviso in N*N sotto matrici, le matrici sulla diagonale come reservoir, quindi sono quadrate e 
vengono scalate tramite il raggio spettrale.
Mentre le off-diagonali non sono quadrate.

