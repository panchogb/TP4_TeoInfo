import sys
import os
import math
import numpy as np
import random

def LeerArgumentos():
    args = sys.argv
    cant_arg = len(args)
    if (cant_arg == 4 or cant_arg == 5):
        N = int(args[2])
        M = int(args[3])

        flag = None
        if (cant_arg == 5):
            flag = args[4]
        if (os.path.exists(args[1]) and N >= 0 and M >= 0 and (flag == None or flag == '-p')):
            return True, args[1], N, M, (flag != None)
    return False, None, None, None, None

def Leer_Archivo(dir_archivo):
    prob_fuente = np.zeros(2)
    matriz_canal = np.zeros((2,2))

    with open(dir_archivo, 'r') as archivo:
        for i, linea in enumerate(archivo):
            numeros = linea.split()
            if (i == 0):
                for j, num in enumerate(numeros):
                    prob_fuente[j] = float(num)
            else:                
                for j, num in enumerate(numeros):
                    matriz_canal[i-1,j] = float(num)

    archivo.close()

    return prob_fuente, matriz_canal
###################################################################################
##########################   CALCULOS DE DATOS   ##################################
###################################################################################
def CalcularEntropiaPriori(prob_fuente):
    Entropia_a_priori = 0.0
    for i in range(prob_fuente.shape[0]):
        Entropia_a_priori += prob_fuente[i] * np.log2(1.0/(prob_fuente[i] if (prob_fuente[i] != 0) else 1.0))
    return Entropia_a_priori
def CalcularEntropiaPosteriori(prob_fuente, matriz_canal, j):
    entropia = 0    
    for i in range(prob_fuente.shape[0]):
        bj = 0
        for k in range(prob_fuente.shape[0]):
            bj += prob_fuente[k] * matriz_canal[k, j]
        p_ai_bj = matriz_canal[i, j] * prob_fuente[i] / (bj if (bj != 0) else 1.0)
        entropia += (p_ai_bj * np.log2(1.0/p_ai_bj)) if p_ai_bj != 0 else 0
    return entropia
def CalcularEntropiaMediaPriori(prob_fuente, matriz_canal):
    entropia = 0
    for i in range(prob_fuente.shape[0]):
        HBaj = 0.0
        for j in range(matriz_canal.shape[0]):
            HBaj += (matriz_canal[i, j] * np.log2(1.0/matriz_canal[i, j])) if matriz_canal[i, j] != 0 else 0
        entropia += prob_fuente[i] * HBaj
    return entropia
def CalcularEntropiaMediaPosteriori(prob_fuente, matriz_canal):
    entropia = 0    
    for i in range(prob_fuente.shape[0]):
        b = 0.0
        for k in range(prob_fuente.shape[0]):
            b += prob_fuente[k] * matriz_canal[k, i]
        entropia += b * CalcularEntropiaPosteriori(prob_fuente, matriz_canal, i)
    return entropia
def CalcularInformacionMutua(prob_fuente, matriz_canal):
    infoMutua = 0    
    for i in range(prob_fuente.shape[0]):
        infoMutua += prob_fuente[i] * np.log2(1.0/(prob_fuente[i] if (prob_fuente[i] != 0) else 1.0))
    return infoMutua - CalcularEntropiaMediaPosteriori(prob_fuente, matriz_canal)
def CalcularInformacionMutuaPriori(prob_fuente, matriz_canal):
    HB = 0
    for j in range(matriz_canal.shape[0]):
        Pbi = 0
        for i in range(matriz_canal.shape[0]):
            Pbi += matriz_canal[i, j] * prob_fuente[i]
        HB += Pbi * np.log2(1.0/(Pbi if (Pbi != 0) else 1.0))
    return HB - CalcularEntropiaMediaPriori(prob_fuente, matriz_canal)
def CalcularValores(prob_fuente, matriz_canal):
    print('Datos:')
    print(f'H(A): {CalcularEntropiaPriori(prob_fuente):.4f}')
    for j in range(matriz_canal.shape[0]):
        print(f'H(A/b{j}): {CalcularEntropiaPosteriori(prob_fuente, matriz_canal, j):.4f}')
    print(f'H(A/B): {CalcularEntropiaMediaPosteriori(prob_fuente, matriz_canal):.4f}')
    print(f'H(B/A): {CalcularEntropiaMediaPriori(prob_fuente, matriz_canal):.4f}')
    print(f'I(A/B): {CalcularInformacionMutua(prob_fuente, matriz_canal):.4f}')
    print(f'I(B/A): {CalcularInformacionMutuaPriori(prob_fuente, matriz_canal):.4f}')
###################################################################################
###################################################################################
###################################################################################
def CrearMensaje(prob_fuente, N, M, pc):
    # M: Longitud del mensaje
    # N: Número de mensajes
    matriz = np.random.choice(len(prob_fuente), size=(N, M), p=prob_fuente)
    if (pc):
        matriz = np.pad(matriz, ((0, 1), (0, 1)), mode='constant', constant_values=0)
        VRC = np.sum(matriz, axis=1) % 2
        LRC = np.sum(matriz, axis=0) % 2
        matriz[:,-1] = VRC
        matriz[-1,:] = LRC
        matriz[-1,-1] = (np.sum(VRC, axis=0) + np.sum(LRC, axis=0)) % 2
    return matriz

def SimularEnvioMensaje(mensaje, matriz_canal):
    N, M = mensaje.shape
    mensaje_enviado = np.zeros((N,M), dtype=int)
    for i in range(N):
        for j in range(M):
            mensaje_enviado[i, j] = 0 if random.uniform(0, 1) < matriz_canal[mensaje[i,j], 0] else 1
    return mensaje_enviado


def DetectarErrores(mensaje_recibido, N, M):
    cant_corregidos = 0
    if (mensaje_recibido[-1,-1] == 0):  
        LRC = mensaje_recibido[-1,0:-1]
        VRC = mensaje_recibido[0:-1,-1]

        paridad = (np.sum(LRC) + np.sum(VRC)) % 2
        if (paridad == 0):
            sumL = np.sum(mensaje_recibido[:-1,:-1], axis=0) % 2
            sumV = np.sum(mensaje_recibido[:-1,:-1], axis=1) % 2

            LRC_indices = np.nonzero(sumL != LRC)[0]
            VRC_indices = np.nonzero(sumV != VRC)[0]

            if (VRC_indices.shape[0] == 1 and LRC_indices.shape[0] == 1):
                c = LRC_indices[0]
                f = VRC_indices[0]
                cant_corregidos = 1
                mensaje_recibido[f,c] = np.abs(mensaje_recibido[f,c]-1)
                print(f'\nMensaje corregido:\n{mensaje_recibido}')
                print(f'Error corregido en la columna: {c+1} y fila: {f+1}')
            else:
                if (VRC_indices.shape[0] == 0 and LRC_indices.shape[0] == 0):
                    print(f'\nNo hay errores')
                else:
                    print(f'\nNo se puede corregir, mas de un error')
        else:
            print('\nError en Bits de LRC y VRC')
    else:
        print(f'\nError en bit de paridad curzada')
    print(f'\nCantidad de mensajes corregidos: {cant_corregidos}')

def CompararMensajes(mensaje, mensaje_enviado, N, M): 
    mensajes_correctos = 0
    print('\nComparacion mensajes:')
    for i in range(N):
        j = 0
        while (j < M and mensaje[i, j] == mensaje_enviado[i, j]):
            j += 1
            
        print(f'\n{mensaje_enviado[i, :-1]}')
        print(mensaje[i, :-1], end='')
        if (j >= M):
            mensajes_correctos += 1
            print(' Correcto')
        else:            
            print(' Incorrecto')

    print(f'\nCantidad de mensajes correctos: {mensajes_correctos}')
    print(f'Cantidad de mensajes incorrectos: {N - mensajes_correctos}')


condicion, dir_archivo, N, M, pc = LeerArgumentos() #pc = paridad_cruzada

if (condicion):
    prob_fuente, matriz_canal = Leer_Archivo(dir_archivo)
    CalcularValores(prob_fuente, matriz_canal)
    mensaje = CrearMensaje(prob_fuente, N, M, pc)

    mensaje_enviado = SimularEnvioMensaje(mensaje, matriz_canal)

    print(f'\nMensaje:\n{mensaje}')
    print(f'\nMensaje Enviado:\n{mensaje_enviado}')

    CompararMensajes(mensaje_enviado, mensaje, N, M)

    if (pc):
        DetectarErrores(mensaje_enviado, N, M)
else:
    print('Error de argumentos')