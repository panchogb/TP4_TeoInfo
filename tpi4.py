import sys
import os
import math
import numpy as np
import random

def LeerArgumentos():
    #args = sys.argv
    args = ['tpi4.py', 'tp4_sample7.txt', '3', '7', '-p']
    #
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
        Entropia_a_priori += prob_fuente[i] * np.log2(1.0/prob_fuente[i])
    return Entropia_a_priori
def CalcularEntropiaPosteriori(prob_fuente, matriz_canal, j):
    entropia = 0    
    for i in range(prob_fuente.shape[0]):
        bj = 0
        for k in range(prob_fuente.shape[0]):
            bj += prob_fuente[k] * matriz_canal[k, j]
        p_ai_bj = matriz_canal[i, j] * prob_fuente[i] / bj
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
        infoMutua += prob_fuente[i] * np.log2(1.0/prob_fuente[i])
    return infoMutua - CalcularEntropiaMediaPosteriori(prob_fuente, matriz_canal)
def CalcularInformacionMutuaPriori(prob_fuente, matriz_canal):
    HB = 0
    for j in range(matriz_canal.shape[0]):
        Pbi = 0
        for i in range(matriz_canal.shape[0]):
            Pbi += matriz_canal[i, j] * prob_fuente[i]
        HB += Pbi * np.log2(1.0/Pbi)
    return HB - CalcularEntropiaMediaPriori(prob_fuente, matriz_canal)
def CalcularValores(prob_fuente, matriz_canal):

    print(f'H(A) (Entropia "a-priori" de A): {CalcularEntropiaPriori(prob_fuente):.4f}')
    for j in range(matriz_canal.shape[0]):
        print(f'H(A/b{j}) (Entropia "a-posteriori" de A, recibido b{j}): {CalcularEntropiaPosteriori(prob_fuente, matriz_canal, j):.4f}')
    print(f'H(A/B) (Equivocacion): {CalcularEntropiaMediaPosteriori(prob_fuente, matriz_canal):.4f}')
    print(f'H(B/A): {CalcularEntropiaMediaPriori(prob_fuente, matriz_canal):.4f}')
    print(f'I(A/B) (Informacion mutua): {CalcularInformacionMutua(prob_fuente, matriz_canal):.4f}')
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
        VRC = np.sum(matriz, axis=0) % 2
        LRC = np.sum(matriz, axis=1) % 2
        matriz[-1,:] = VRC
        matriz[:,-1] = LRC
        matriz[-1,-1] = (np.sum(VRC, axis=0) + np.sum(LRC, axis=0)) % 2
    return matriz

def SimularEnvioMensaje(mensaje, matriz_canal, N, M, pc):
    mensaje_enviado = np.copy(mensaje)
    for i in range(N):
        for j in range(M):
            mensaje_enviado[i, j] = 0 if random.uniform(0, 1) < matriz_canal[mensaje[i,j], 0] else 1
    return mensaje_enviado


def DetectarErrores(mensaje_recibido, N, M):    
    sumV = np.sum(mensaje_recibido[:-1,:-1], axis=0) % 2
    sumL = np.sum(mensaje_recibido[:-1,:-1], axis=1) % 2

    VRC = mensaje_recibido[-1,0:-1]
    LRC = mensaje_recibido[0:-1,-1]

    VRC_indices = np.nonzero(sumV != VRC)[0]
    LRC_indices = np.nonzero(sumL != LRC)[0]

    if (VRC_indices.shape[0] == 1 and LRC_indices.shape[0] == 1):
        c = VRC_indices[0]
        f = LRC_indices[0]
        print(f'Se puede corregir, error en columna: {c} y fila: {f}')
        mensaje_recibido[f,c] = np.abs(mensaje_recibido[f,c]-1)
        print(f'Mensaje corregido:\n{mensaje_recibido}')
    else:
        print(f'No se puede corregir')

condicion, dir_archivo, N, M, pc = LeerArgumentos() #pc = paridad_cruzada

if (condicion):
    prob_fuente, matriz_canal = Leer_Archivo(dir_archivo)
    CalcularValores(prob_fuente, matriz_canal)
    mensaje = CrearMensaje(prob_fuente, N, M, pc)
    mensaje_enviado = SimularEnvioMensaje(mensaje, matriz_canal, N, M, pc)

    print(f'Mensaje:\n{mensaje}')
    print(f'Mensaje Enviado:\n{mensaje_enviado}')

    if (pc):
        DetectarErrores(mensaje_enviado, N, M)
else:
    print('Error de argumentos')