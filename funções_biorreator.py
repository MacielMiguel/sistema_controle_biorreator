''' -------- BIBLIOTECAS UTILIZADAS -------- '''

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import control as ct
from geneticalgorithm import geneticalgorithm as ga 
from do_mpc.model import Model
from do_mpc.controller import MPC
from do_mpc.simulator import Simulator
from do_mpc.graphics import Graphics
from scipy.integrate import solve_ivp


''' -------- CLASSES IMPORTANTES -------- '''

class PID:
    """
    Implementação de um controlador PID (Proporcional-Integral-Derivativo).

    Esta classe calcula um valor de saída para corrigir um erro medido,
    com o objetivo de levar o sistema a um setpoint desejado.
    Conta com limite de saída de atuadores e anti-windup

    Atributos:
        Kp (float): Ganho do termo Proporcional.
        Ki (float): Ganho do termo Integral.
        Kd (float): Ganho do termo Derivativo.
        output_limits (Tuple[Optional[float], Optional[float]]): Limites de saída do atuador.
    """

    def __init__(self, Kp, Ki, Kd, output_limits=(None, None), sample_time=0.01):
        """
        Inicializa o controlador PID.

        Args:
            Kp (float): O ganho do termo Proporcional. Define a reação ao erro atual.
            Ki (float): O ganho do termo Integral. Elimina o erro em regime estacionário
                        somando os erros passados.
            Kd (float): O ganho do termo Derivativo. Responde à taxa de variação do erro,
                        ajudando a prever o comportamento futuro e a amortecer a resposta.
            output_limits (Tuple[Optional[float], Optional[float]], optional):
                        Uma tupla contendo os limites mínimo e máximo da saída.
                        Usado para prevenir saturação do atuador (anti-windup).
                        Por padrão é (None, None), sem limites.
            sample_time (float, optional): O tempo mínimo de amostragem em segundos
                                           entre os cálculos. Evita que o controlador
                                           reaja a flutuações muito rápidas (ruído).
                                           Por padrão é 0.01 segundos.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self._min_output, self._max_output = output_limits

        self._sample_time = sample_time
        self._last_time = None

        self._proportional = 0
        self._integral = 0
        self._derivative = 0

        self._previous_error = 0 # Erro do ciclo anterior para o termo derivativo
        self._last_output = 0    # Última saída calculada

        self.reset()

    def update(self, current_error, current_time=None):
        """
        Calcula a nova saída do controlador com base no erro atual.

        Esta função deve ser chamada periodicamente dentro do loop de controle.

        Args:
            current_error (float): O erro atual do sistema (setpoint - valor_medido).
            current_time (float): O tempo atual em segundos. É crucial para calcular
                                  a variação de tempo (dt) para os termos I e D.

        Returns:
            float: O valor de controle calculado. Este valor permanece dentro dos
                   `output_limits`, se definidos.
        """
        if self._last_time is None:
            self._last_time = current_time
            return 0.0  # Retorna 0 na primeira chamada para evitar cálculos com dt inválido

        dt = current_time - self._last_time

        if dt >= self._sample_time:
            # Termo Proporcional
            self._proportional = current_error

            # Termo Integral
            self._integral += current_error * dt
            # Anti-windup (limitando o termo integral para evitar saturação excessiva)
            if self._min_output is not None and self._max_output is not None:
                if self._integral * self.Ki > self._max_output:
                    self._integral = self._max_output / self.Ki
                elif self._integral * self.Ki < self._min_output:
                    self._integral = self._min_output / self.Ki
            
            # Termo Derivativo
            # Evitar picos no derivativo se o erro mudar muito rapidamente por causa de ruído
            self._derivative = (current_error - self._previous_error) / dt if dt > 0 else 0

            # Calcula a saída total
            output = (self.Kp * self._proportional +
                      self.Ki * self._integral +
                      self.Kd * self._derivative)

            # Aplica limites na saída
            if self._min_output is not None:
                output = max(output, self._min_output)
            if self._max_output is not None:
                output = min(output, self._max_output)

            self._last_output = output
            self._previous_error = current_error
            self._last_time = current_time
        
        return self._last_output

    def set_tunings(self, Kp, Ki, Kd):
        """
        Permite ajustar os ganhos do controlador em tempo de execução.

        Args:
            Kp (float): O novo ganho Proporcional.
            Ki (float): O novo ganho Integral.
            Kd (float): O novo ganho Derivativo.
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

    def set_output_limits(self, min_output, max_output):
        """
        Define os limites mínimo e máximo da saída do controlador.

        Args:
            min_val (Optional[float]): O limite inferior da saída.
            max_val (Optional[float]): O limite superior da saída.
        """
        self._min_output = min_output
        self._max_output = max_output
        # Resetar integral para evitar wind-up se os limites mudarem
        self.reset()

    def reset(self):
        """
        Reinicia o estado interno do controlador PID.

        Zera todos os termos (P, I, D) e o histórico de erro, útil para quando
        o controle é reativado.
        """

        self._proportional = 0
        self._integral = 0
        self._derivative = 0
        self._previous_error = 0
        self._last_time = None # Força o cálculo de um novo dt na próxima atualização
        self._last_output = 0

class BioreactorSystem:
    """
    Simula o comportamento de um sistema de biorreator de cultura contínua.

    Esta classe encapsula o modelo matemático de um quimiostato, utilizando
    a cinética de Monod para descrever o crescimento celular. Ela calcula a
    evolução das concentrações de biomassa (X) e substrato (S) ao longo do tempo,
    dadas as taxas de diluição (D) e a concentração de substrato na alimentação (Sf).

    Atributos:
        D (float): A taxa de diluição atual (h⁻¹).
        Sf (float): A concentração de substrato na alimentação atual (g/L).
        mi_max (float): A taxa de crescimento específico máxima (h⁻¹).
        ks (float): A constante de saturação de Monod (g/L).
        Y (float): O coeficiente de rendimento biomassa/substrato (g de biomassa / g de substrato).
    """
    def __init__(self, initial_D, initial_Sf, x0, s0):
        """
        Inicializa o sistema do biorreator com os seus parâmetros e estado inicial.

        Args:
            initial_D (float): A taxa de diluição inicial (h⁻¹).
            initial_Sf (float): A concentração inicial de substrato na alimentação (g/L).
            x0 (float): A concentração inicial de biomassa (g/L).
            s0 (float): A concentração inicial de substrato no reator (g/L).
        """
        self.D = initial_D
        self.Sf = initial_Sf
        self._measured_x = x0
        self._measured_s = s0
        self.mi_max = 0.5    # (L/h)
        self.ks = 0.1    # (g/L)
        self.Y = 0.4    # (g/g)
        self.mi = 0.0
        self._last_time = 0.0
        self._current_time = 0.0

    def _mi_function(self, S):
        """
        Calcula a taxa de crescimento específico (mi) usando a equação de Monod.

        Args:
            S (float): A concentração atual de substrato (g/L).

        Returns:
            float: A taxa de crescimento específico calculada (h⁻¹).
        """
        return (self.mi_max * S)/(self.ks + S)

    def _derivatives(self, y, t, D, Sf):
        """
        Define o sistema de equações diferenciais ordinárias (EDOs) para o biorreator.
        Esta função é formatada para ser usada com o solver `scipy.integrate.odeint`.

        Args:
            y (List[float]): Um vetor contendo os valores atuais do estado [X, S].
            t (float): O ponto de tempo atual (não usado diretamente, mas exigido pelo solver).
            D (float): A taxa de diluição atual (h⁻¹).
            Sf (float): A concentração de substrato na alimentação atual (g/L).

        Returns:
            np.ndarray: Um array contendo as derivadas [dX/dt, dS/dt].
        """
        X, S = y

        mi = self._mi_function(S)

        dxdt = mi * X - D * X
        dsdt = D * (Sf - S) - mi * (1 / self.Y) * X

        return np.array([dxdt, dsdt])

    def update_system(self, D_pid, Sf_pid, dt_simulation):
        """
        Avança a simulação do sistema por um intervalo de tempo `dt_simulation`.
        Utiliza um solver de EDOs para integrar as equações do sistema e atualizar
        as concentrações de biomassa e substrato.

        Args:
            D_pid (float): O novo valor da taxa de diluição, tipicamente vindo de um controlador.
            Sf_pid (float): O novo valor da concentração de alimentação, vindo de um controlador.
            dt_simulation (float): O intervalo de tempo da simulação (h).  

        Returns:
            np.ndarray: Um array com o novo estado do sistema [measured_x, measured_s].
        """
        self.D = D_pid
        self.Sf = Sf_pid

        time = [self._last_time, self._last_time + dt_simulation]
        y0 = [self._measured_x, self._measured_s]
        resp = odeint(
                        self._derivatives,
                        y0=y0,
                        t=time,
                        args=(self.D, self.Sf)
                      )
        
        self._measured_x, self._measured_s = resp[-1]
        self._last_time += dt_simulation

        return np.array([self._measured_x, self._measured_s])
    
    def reset(self):
        """
        Reinicia o sistema para um estado inicial predefinido.
        """
        self.D = 0.17
        self.Sf = 1.0
        self._measured_x = 0.38
        self._measured_s = 0.05
        self._last_time = 0.0
        self._current_time = 0.0


''' -------- FUNÇÕES UTILIZADAS DURANTE O CÓDIGO -------- '''

# Constantes das equações de comportamento do sistema
Sf = 1    # (L/h)
mi_max = 0.5    # (L/h)
Ks = 0.1    # (g/L)
Y = 0.4    # (g/g)
D_max = (mi_max*Sf)/(Ks + Sf)    # (1/h)

def mi_function(S):    # (L/h)
    '''
    Função para cálculo do valor de mi dado um valor de S.

    Args:
        S (float): valor de susbtrato atual do sistema.

    Returns:
        float: valor de mi atual do sistema.
    '''
    return (mi_max * S)/(Ks + S)

def X_derivative(S, X, D):
    '''
    Função para cálculo do valor da derivada de X
    dados os valores de X, S e D atuais.

    Args:
        X (float): valor de biomassa atual do sistema.
        S (float): valor de susbtrato atual do sistema.
        D (float): valor de fluxo atual do sistema.

    Returns:
        float: valor da derivada de X atual do sistema.
    '''
    return mi_function(S)*X - D*X

def S_derivative(Sf, S, X, D):
    '''
    Função para cálculo do valor da derivada de S
    dados os valores de Sf, S, X e D atuais.

    Args:
        Sf (float): valor de susbtrato de entrada atual do sistema.
        S (float): valor de susbtrato atual do sistema.
        X (float): valor de biomassa atual do sistema.
        D (float): valor de fluxo atual do sistema.

    Returns:
        float: valor da derivada de S atual do sistema.
    '''
    return D*(Sf - S) - mi_function(S)*(1/Y)*X

# Funções para simulação do sistema não linear através da odeint
def bioreactor_nonlinear(y, t, D_func, Sf_func, X_derivative, S_derivative):
    '''
    Função para cálculo do valor das derivadas do sistema no instante de tempo passado.

    Args:
        y (float): valores de X e S para a seguinte iteração.
        t (float): valor atual de tempo para o cálculo do comportamento do sistema.
        D_func (Callable[float]): função de cálculo do valor de D para o instante t.
        Sf_func (Callable[float]): função de cálculo do valor de Sf para o instante t.
        X_derivative (Callable[float, float, Callable[float]]): função para o cálculo da derivada
                                                                de X no instante de tempo passado
                                                                segundo as equações do sistema.
        S_derivative (Callable[Callable[float], float, float, float]): função para o cálculo da derivada
                                                                de S no instante de tempo passado
                                                                segundo as equações do sistema.

    Returns:
        NDArray[np.float64]: com os valores de dX/dt e dS/dt.
    '''
    x, s = y
    D_value = D_func(t)
    Sf_value = Sf_func(t)
    
    dxdt = X_derivative(S=s, X=x, D=D_value)
    dsdt = S_derivative(Sf=Sf_value, S=s, X=x, D=D_value)
    return np.array([dxdt, dsdt])

def non_linear_simulation(x0, s0, time, D_input, Sf_input):
    ''' 
    Função para simular o comportamento do sistema a partir de certas entradas.

    Args:
        x0 (float): valor inicial de X
        s0 (float): valor inicial de S
        time (NDArray[np.float64]): conjunto de instantes de tempo p/ cálculo do comportamento do sistema.
        D_input (Callable[float]): função p/ determinação do valor de D no instante passado.
        Sf_input (Callable[float]): função p/ determinação do valor de Sf no instante passado.
    
    Returns:
        NDArray[np.float64]: conjunto de valores de X ao longo do tempo informado.
        NDArray[np.float64]: conjunto de valores de S ao longo do tempo informado.
    '''
    D_in = D_input
    Sf_in = Sf_input
    x0=x0
    s0=s0

    # Resolução das equações diferenciais simples (primeira/segunda ordem)
    result = odeint(bioreactor_nonlinear, [x0, s0], time, args=(D_in, Sf_in))
    x, s = result.T

    return x, s

def linearizar_sistema_espaco_estado(D_ss, Sf_ss):
    '''
    Função para linearização do sistema do biorreator no steady state relativo aos valores de D (fluxo)
    e Sf (Substrato de entrada) passados.

    Args:
        D_ss (float): valor de D (fluxo) constante para cálculo do steady state.
        Sf_ss (float): valor de Sf (fluxo) constante para cálculo do steady state.
    
    Returns:
        control.StateSpace: um objeto da classe control.ss representando a linearização do sistema
                            no espaço de estados.
    '''
    # Constantes padrão do sistema
    MI_MAX = 0.5
    K_S = 0.1
    Y = 0.4

    # Ponto de Operação obtido das equações 7 e 8 no artigo
    S_ss = K_S * D_ss / (MI_MAX - D_ss)
    X_ss = Y * (Sf_ss - S_ss)
    mi_ss = (MI_MAX * S_ss)/(K_S + S_ss)

    print(f"Ponto de Operação Estacionário:")
    print(f"D_ss = {D_ss:.2f} L/h, Sf_ss = {Sf_ss:.2f} g/L")
    print(f"X_ss = {X_ss:.4f} g/L, S_ss = {S_ss:.4f} g/L")

    # Matrizes do Sistema Linearizado (Resultados obtidos manualmente)
    a = (MI_MAX * K_S) / (K_S + S_ss)**2
    A00 = mi_ss - D_ss
    A01 = X_ss * a
    A10 = -mi_ss/Y
    A11 = -D_ss - (X_ss * (a/Y))

    A = np.array([[A00, A01],
                [A10, A11]])

    B = np.array([[-X_ss, 0],
                [Sf_ss - S_ss, D_ss]])

    C = np.eye(2)

    D = np.zeros((2, 2))

    # Definição do sistema linearizado para configuração passada
    sys_linear = ct.ss(A, B, C, D)

    return sys_linear # Retorno de sistema linearizado no espaço de estados

def calcular_matriz_rga(sys_linear):
    '''
    Função para o cálculo da matrix RGA do sistema a partir de sua linearização
    no espaço de estados.

    Args:
        sys_linear (control.StateSpace): sistema linearizado no espaço de estados.
        
    Returns:
        NDArray[np.float64]: matriz RGA obtida.
    '''
    sys_tf = ct.tf(sys_linear)

    k = np.asarray(ct.dcgain(sys_tf), dtype=float)
    
    k_inv = np.linalg.inv(k)

    rga_matriz = k * k_inv.T 

    return rga_matriz

def get_parameters_znrc(y, amp_degrau, deg_tempo, time):
    '''
    Função para cálculo de parâmetros do sistema a partir da curva de reação (Ziegler Nichols Reaction Curve).

    Args:
        y (NDArray[np.float64]): valores a variável manipulada de análise em questão.
        amp_degrau (float): diferença entre a ação de controle inicial e final.
        deg_tempo (float): instante de tempo em que o degrau é aplicado.
        time (NDArray[np.float64]): conjunto de instantes de tempo em que o sistema foi avaliado.
    
    Returns:
        float: valor do ganho do sistema.
        float: valor da constante de tempo do sistema.
        float: valor do atraso do sistema.
    '''
    y_initial = y[np.where(time >= deg_tempo)[0][0]]
    delta = y[-1] - y_initial    # Supõe que o sistema chegou no regime estacionário

    # Determinação de Kp para análise da resposta
    K = (delta)/amp_degrau
    
    # Cálculo do theta
    theta = 0.0
    threshold = 0.02    # 2%
    y_condition = y_initial + threshold*delta
    indices = np.where(y >= y_condition)[0]
    theta = time[indices[0]] - deg_tempo

    # Cálculo de tau
    tau = 0.0
    y_initial = y[indices[0]]
    indices_tau = np.where(y >= (y_initial + 0.632*delta))[0]
    tau = time[indices_tau[0]] - theta

    return K, tau, theta

# Função para sintonia de valores de ganhos do PID por Ziegler-Nichols reaction curve
def znrc_sintonia(Kp, tau, theta, tipo_de_controlador='PI'):
    '''
    Função para sintonizar os valores de Kp, Ki e Kd a paetir das relações
    de Zigler-Nichols para os tipos de controlador: P (Proporcional), PI(Proporcional-Integrativo)
    e PID(Proporcional-Integrativo-Derivativo).

    Args:
        Kp (float): ganho do processo (delta_y/delta_u)
        tau (float): constante de tempo do processo
        theta (float): atraso do processo
        tipo_de_controlador (String)
    '''

    if tipo_de_controlador == 'P':
        Kp = tau / (theta)
        Ti = float('inf')
        Td = 1e-16
    elif tipo_de_controlador == 'PI':
        Kp = 0.9 * tau / (theta)
        Ti = tau / 0.3
        Td = 0.0
    elif tipo_de_controlador == 'PID':
        Kp = 1.2 * tau / (theta)
        Ti = 2.0 * theta
        Td = 0.5 * theta

    Ki = Kp / Ti
    Kd = Kp * Td
    return np.array([Kp, Ki, Kd])

def imc_sintonia(Kp, tau, theta, lambda_imc):
    '''
    Função para sintonia a partir do método do IMC considerando o modelo aproximado como de primeira ordem
    com atraso.

    Args:
        Kp (float): valor do ganho do sistema.
        tau (float): valor da cosntante de tempo do sistema.
        theta (float): valor do atraso do sistema.
        lambda_imc (float): parâmetro regulador de abruptividade de controle do sistema.

    Returns:
        Kp (float): valor do ganho proporcional do sistema.
        Ki (float): valor do ganho integrativo do sistema.
        Kd (float): valor do ganho derivativo do sistema.
    '''
    # Determinação dos parâmetros baseado nas relações de IMC p/ modelo de primeira ordem
    Kc = tau/(Kp * (lambda_imc + theta))
    Ti = tau
    Td = 0.0

    # Determinação dos parâmetros de sintonia
    Kp = Kc
    Ki = Kc/Ti
    Kd = Kc * Td

    return Kp, Ki, Kd

''' FUNÇÃO DO MÉTODO DA OSCILAÇÃO SUSTENTADA DE ZN'''

''' FUNÇÃO DO MÉTODO CASCATA '''

# Função para extração de parâmetros de performances dos gráficos
def param_performance(setpoint, y, time):
    '''
    Função para extração de parâmetros de performance da resposta do sistema a um degrau na entrada.
    Determina parâmetros como: ISE (Integral Squared Error/Erro Quadrado Integral),
    tempo se subida, tempo de acomodação e percentual de sobressinal. 

    Args:
        setpoint (float): valor em que se deseja chegar na variável manipuladas
        y (np.array[dtype=float]): resposta do sistema para um das variáveis manipuladas
        time (np.array[dtype=float]): conjunto de pontos de tempo utilizados no sistema

    Returns:
        Dict: dicioonário com os valores de performance todos relacionados com seus nomes.
    '''

    metrics = {}
    y = np.array(y)
    dt = time[1]
    
    # Cálculo do erro de regime permamente (supondo que o sistema tenha chegado nele)
    y_ss = y[-1]
    metrics['SSE'] = setpoint - y_ss

    # Erro Quadrático Integrado (Integral Squared Error - ISE)
    errors = setpoint - y    # Erro em cada ponto do tempo
    metrics['ISE'] = np.sum(errors**2) * dt

    # Performance na resposta ao degrau
    delta = setpoint - y[0]

    # Tempo de Subida, tempo para ir de 10% a 90% da mudança total
    rise_time_10_percent_val = 0.1 * y_ss
    rise_time_90_percent_val = 0.9 * y_ss

    # Encontra os índices onde a resposta cruza os limites de 10% e 90%
    idx_10 = np.where(y >= rise_time_10_percent_val)[0]
    idx_90 = np.where(y >= rise_time_90_percent_val)[0]

    if len(idx_10) > 0 and len(idx_90) > 0:
        t_10 = time[idx_10[0]]
        t_90 = time[idx_90[0]]
        metrics['rise_time'] = t_90 - t_10
    else:
        metrics['rise_time'] = np.nan    # Não foi possível calcular

    # Sobressinal
    max_y = np.max(y)
    if max_y > setpoint:
        # Porcentagem em relação à mudança total do degrau
        metrics['overshoot'] = ((max_y - setpoint) / setpoint) * 100
    else:
        metrics['overshoot'] = 0.0 # Não houve ultrapassagem

    # Tempo de Acomodação 
    band_percentage = 0.02    # Margem de erro de 2% para cima e para baixo
    settling_upper_bound = setpoint * (1 + band_percentage)
    settling_lower_bound = setpoint * (1 - band_percentage)
    
    
    if delta >= 0:    # Se a resposta vai para cima (degrau positivo)
        # Encontra o último ponto em que a resposta está fora da banda superior
        outside_band_upper_idx = np.where(y > settling_upper_bound)[0]
        # Encontra o último ponto em que a resposta está fora da banda inferior
        outside_band_lower_idx = np.where(y < settling_lower_bound)[0]
    else:    # Se a resposta vai para baixo (degrau negativo)
        outside_band_upper_idx = np.where(y > settling_lower_bound)[0]    # Invertido para decrescente
        outside_band_lower_idx = np.where(y < settling_upper_bound)[0]    # Invertido para decrescente

    last_outside_idx_upper = outside_band_upper_idx[-1] if len(outside_band_upper_idx) > 0 else 0
    last_outside_idx_lower = outside_band_lower_idx[-1] if len(outside_band_lower_idx) > 0 else 0

    settling_time_index = max(last_outside_idx_upper, last_outside_idx_lower)

    metrics['settling_time'] = time[settling_time_index]
    return metrics

def simula_sistema_pid_desacoplado(sp_x, sp_s, biorreator, pid_x, pid_s, time):
    '''
    Função para simulação do sistema do biorreator PIDs desacoplados. Cálcula a cada iteração
    o valor das ações de controle e a mudança que gera no sistema.

    Args:
        sp_x (float): valor do setpoint para X.
        sp_s (float): valor do setpoint para S.
        biorreator (BioreatorSystem): sistema do biorreator com seus parâmetros de entrada inicial já definidos.
        pid_x (PID): objeto PID já sintonizado para cálculo da ação de controle de X.
        pid_s (PID): objeto PID já sintonizado para cálculo da ação de controle de S.
    
    Returns:
        NDArray[np.float64]: matriz com os valores de X na primeira linha e as ações de controle na segunda. 
        NDArray[np.float64]: matriz com os valores de S na primeira linha e as ações de controle na segunda.
    '''
    # Esquecendo possíveis outras simulações
    biorreator.reset()
    pid_x.reset()
    pid_s.reset()

    # Criação dos arrays para plot contendo uma linha para os valores das variáveis manipuladas
    # e uma linha para as ações de controle
    linhas = 2
    colunas = len(time)
    x_plot = np.zeros((linhas, colunas), dtype=float)
    s_plot = np.zeros((linhas, colunas), dtype=float)
    current_values = np.array([0.0, 0.0])
    indice = 0
    for i in time:
        current_time = i

        # Calcular o erro
        error_x = sp_x - current_values[0] # Erro em X => g/L de X desejado - g/L de X medido
        error_s = sp_s - current_values[1] # Erro em S => g/L de S desejado - g/L de S medido

        # Atualizar o PID e obter a saída de controle
        pid_x_output = pid_x.update(error_x, current_time=current_time)
        pid_s_output = pid_s.update(error_s, current_time=current_time)
        x_plot[1][indice] = pid_x_output
        s_plot[1][indice] = pid_s_output

        # Atualiza os valores das variáveis manipuladas
        current_values = biorreator.update_system(pid_s_output, pid_x_output, time[1])
        x_plot[0][indice] = current_values[0]
        s_plot[0][indice] = current_values[1]
        indice += 1


    print(f"Simulação concluída para X. Valor final do sistema: {current_values[0]:.2f}")
    print(f"Simulação concluída para S. Valor final do sistema: {current_values[1]:.2f}")
    return x_plot, s_plot

def simula_sistema_pid_desacoplado_com_disturbio(sp_x, sp_s, biorreator, pid_x, pid_s, time, dist_ac=True):
    '''
    Função para simulação do sistema do biorreator PIDs desacoplados com distúrbios nas ações de controle
    ou leituras de sensores do sistema. Os instantes de tempo para a inserção dos distúrbios é padronizada. 
    
    Cálcula a cada iteração o valor das ações de controle e a mudança que gera no sistema.

    Args:
        sp_x (float): valor do setpoint para X.
        sp_s (float): valor do setpoint para S.
        biorreator (BioreatorSystem): sistema do biorreator com seus parâmetros de entrada inicial já definidos.
        pid_x (PID): objeto PID já sintonizado para cálculo da ação de controle de X.
        pid_s (PID): objeto PID já sintonizado para cálculo da ação de controle de S.
        dist_ac (boolean): determina se o distúrbio será nas ações de controle ou nas leituras do sistema.
    
    Returns:
        NDArray[np.float64]: matriz com os valores de X na primeira linha e as ações de controle na segunda. 
        NDArray[np.float64]: matriz com os valores de S na primeira linha e as ações de controle na segunda.
    '''
    # Esquecendo possíveis outras simulações
    biorreator.reset()
    pid_x.reset()
    pid_s.reset()

    # Configurando os distúrbios
    dist_control = dist_ac
    dist_1_inicio = 5.5    # Instante em hora do início do primeiro distúrbio
    dist_1_fim = 5.7    # Instante em hora do fim do primeiro distúrbio

    dist_2_inicio = 7.2    # Instante em hora de início do segundo distúrbio
    dist_2_fim = 7.3    # Instante em hora de fim do segundo distúrbio

    # Criação dos arrays para plot contendo uma linha para os valores das variáveis manipuladas
    # e uma linha para as ações de controle
    linhas = 2
    colunas = len(time)
    x_plot = np.zeros((linhas, colunas), dtype=float)
    s_plot = np.zeros((linhas, colunas), dtype=float)
    current_values = np.array([0.0, 0.0])
    indice = 0
    for i in time:
        current_time = i

        # Calcular o erro
        error_x = sp_x - current_values[0] # Erro em X => g/L de X desejado - g/L de X medido
        error_s = sp_s - current_values[1] # Erro em S => g/L de S desejado - g/L de S medido

        # Atualizar o PID e obter a saída de controle
        pid_x_output = pid_x.update(error_x, current_time=current_time)
        pid_s_output = pid_s.update(error_s, current_time=current_time)
        x_plot[1][indice] = pid_x_output
        s_plot[1][indice] = pid_s_output

        # Distúrbios na ação de controle de X (Alteração do valor de Sf)
        # Distúrbio 1
        if dist_control==True and (current_time >= dist_1_inicio and current_time <= dist_1_fim):
            pid_x_output += 0.2 
            current_values = biorreator.update_system(pid_s_output, pid_x_output, time[1])  
        # Distúrbio 2
        elif dist_control==True and (current_time >= dist_2_inicio and current_time <= dist_2_fim):
            pid_x_output -= 0.5
            current_values = biorreator.update_system(pid_s_output, pid_x_output, time[1])
        # Sem distúrbio na ação de controle
        else:
            current_values = biorreator.update_system(pid_s_output, pid_x_output, time[1])

        # Distúrbios na leitura de X
        # Distúrbio 1
        if dist_control == False and (current_time >= dist_1_inicio and current_time <= dist_1_fim):
            current_values[0] += 0.03   
        # Distúrbio 2
        if dist_control == False and (current_time >= dist_2_inicio and current_time <= dist_2_fim):
            current_values[0] -= 0.03    

        # Atualiza os valores das variáveis manipuladas
        current_values = biorreator.update_system(pid_s_output, pid_x_output, time[1])
        x_plot[0][indice] = current_values[0]
        s_plot[0][indice] = current_values[1]
        indice += 1
    
    print(f"Simulação concluída para X. Valor final do sistema: {current_values[0]:.2f}")
    print(f"Simulação concluída para S. Valor final do sistema: {current_values[1]:.2f}")
    return x_plot, s_plot

def plotagem_resultados(x_plot, s_plot, time, acao_controle=False):
    ''' 
    Função padronizada para plotagem dos resultados das simulações dos diferentes PIDs.
    
    Args:
        x_plot (): array dos valores das variáveis manipuladas e das ações de controle de X.
        s_plot (): array dos valores das variáveis manipuladas e das ações de controle de S.
        time (): array dos pontos de tempo do sistema.
        acao_controle (boolean): Define se os gráficos devem ou não ter a ação de controle junto 
                                 p/ comparação do comportamento.
    '''
    plt.figure(figsize=(8, 5))
    plt.plot(time, x_plot[0], color='blue')
    if acao_controle == True:
        plt.title('Gráfico de X e sua ação de controle')
        plt.plot(time, x_plot[1], color='red', linestyle='--')
    else:
        plt.title('Gráfico de X')
    plt.xlabel('Tempo (h)')
    plt.ylabel('X (g/L)')
    plt.grid()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(time, s_plot[0], color='blue')
    if acao_controle == True:
        plt.title('Gráfico de S e sua ação de controle')
        plt.plot(time, s_plot[1], color='red', linestyle='--')
    else:
        plt.title('Gráfico de S')
    plt.xlabel('Tempo (h)')
    plt.ylabel('S (g/L)')
    plt.grid()
    plt.show()