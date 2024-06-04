import numpy as np
import scipy.stats as stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_effectsize
import streamlit as st
import matplotlib.pyplot as plt
# https://chatgpt.com/share/42690081-d654-4562-a808-903c8d372af6

# Función para calcular el tamaño de la muestra
def calculate_sample_size(baseline_rate, min_detectable_effect, alpha=0.05, power=0.8):
    effect_size = proportion_effectsize(baseline_rate, baseline_rate + min_detectable_effect)
    analysis = NormalIndPower()
    sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1)
    return int(np.ceil(sample_size))

# Función para verificar el balance de la muestra
def check_sample_balance(visits_A, visits_B, p=0.5, alpha=0.05):
    n = visits_A + visits_B
    E_A = n*p # valor esperado A
    E_B = n*(1-p) # valor esperado B
    V_A = V_B = n*p*(1-p) # varianza A -> np(1-p)

    E_differences = E_A - E_B
    V_differences = V_A + V_B # formula es Var(A) + Var(B) - 2*Cov(A,B), pero Cov(A,B) se asume como independiente
    
    std_differences = V_differences**(0.5) # raiz de varianza

    upper_interval_differences = E_differences + stats.norm.ppf(1-alpha/2, loc=E_differences, scale=std_differences)*std_differences/(n**0.5)
    lower_interval_differences = E_differences - stats.norm.ppf(1-alpha/2, loc=E_differences, scale=std_differences)*std_differences/(n**0.5)

    # print(upper_interval_differences, lower_interval_differences)
    return (lower_interval_differences, upper_interval_differences)

# Función para analizar el A/B test
def analyze_ab_test(conversions_A, conversions_B, visits_A, visits_B, alpha=0.05):
    # Tasas de conversión
    conversion_rate_A = conversions_A / visits_A
    conversion_rate_B = conversions_B / visits_B
    
    # Prueba Z para proporciones
    counts = np.array([conversions_A, conversions_B])
    nobs = np.array([visits_A, visits_B])
    z_stat, p_value = proportions_ztest(counts, nobs, alternative='two-sided')
    
    # Calcular el poder estadístico
    p_combined = (conversions_A + conversions_B) / (visits_A + visits_B)
    effect_size = proportion_effectsize(p_combined, conversion_rate_B)
    power_analysis = NormalIndPower()
    power = power_analysis.solve_power(effect_size=effect_size, nobs1=visits_A, alpha=alpha, ratio=visits_B/visits_A, alternative='two-sided')
    
    return p_value, power

# Generación de datos simulados
def generate_data(visits, conversion_rate_A, conversion_rate_B):
    conversions_A = np.random.binomial(visits, conversion_rate_A)
    conversions_B = np.random.binomial(visits, conversion_rate_B)
    return conversions_A, conversions_B

# Función para calcular los días necesarios para correr el experimento
def calculate_days(visits_per_day, sample_size):
    days = sample_size / visits_per_day
    return int(np.ceil(days))

# Aplicación Streamlit
def main():
    st.title('SMARTest 2.0')

    tab1, tab2, tab3 = st.tabs(["Pre Test", "Balance de Muestra", "Análisis Post Test"])

    with tab1:
        st.header('Pre Test: Coming soon')
        
        # baseline_rate = st.slider('Tasa de conversión de la Versión A (baseline):', min_value=0.01, max_value=0.5, value=0.05)
        # min_detectable_effect = st.slider('Diferencia mínima detectable:', min_value=0.001, max_value=0.1, value=0.01)
        # alpha = st.slider('Nivel de significancia (alfa):', min_value=0.01, max_value=0.1, value=0.05)
        # power = st.slider('Poder estadístico deseado:', min_value=0.5, max_value=0.99, value=0.8)
        # visits_per_day = st.number_input('Visitas por día:', min_value=100, value=500)
        
        # if st.button('Calcular tamaño de muestra y días necesarios'):
        #     # Cálculo del tamaño de la muestra
        #     sample_size = calculate_sample_size(baseline_rate, min_detectable_effect, alpha, power)
        #     st.write(f'Tamaño de muestra necesario por grupo: {sample_size}')
            
        #     # Cálculo de los días necesarios
        #     days_needed = calculate_days(visits_per_day, sample_size)
        #     st.write(f'Días necesarios para correr el experimento: {days_needed} días')

    with tab2:
        st.header('Balance de la Muestra')
        
        visits_A = st.number_input('Número de visitas en la Versión A:', min_value=100, value=1000)
        visits_B = st.number_input('Número de visitas en la Versión B:', min_value=100, value=1000)
        n = visits_A + visits_B

        if st.button('Verificar balance de la muestra'):
            upper_interval_differences, lower_interval_differences = check_sample_balance(visits_A, visits_B)
            st.write(f'Intervalo de confianza para la diferencia de muestras: {upper_interval_differences, lower_interval_differences}')
            if upper_interval_differences <= visits_A - visits_B <= lower_interval_differences:
                st.success(f'La muestra está balanceada, la diferencia de muestras {visits_A - visits_B} entra dentro del intervalo de confianza.')
            else:
                st.warning(f'La muestra no está balanceada, la diferencia de muestras {visits_A - visits_B} NO entra dentro del intervalo de confianza.')

    with tab3:
        st.header('Análisis Post Test')
        
        visits_A = st.number_input('Número de visitas en la Versión A:', min_value=100, value=1000, key="visits_A_post")
        visits_B = st.number_input('Número de visitas en la Versión B:', min_value=100, value=1000, key="visits_B_post")
        conversions_A = st.number_input('Número de conversiones en la Versión A:', min_value=0, value=50)
        conversions_B = st.number_input('Número de conversiones en la Versión B:', min_value=0, value=60)
        alpha_post = st.slider('Nivel de significancia (alfa):', min_value=0.01, max_value=0.1, value=0.05, key="alpha_post")
        
        if st.button('Analizar resultados del A/B Test'):
            p_value, power_observed = analyze_ab_test(conversions_A, conversions_B, visits_A, visits_B, alpha_post)
            st.write(f'P-Valor: {p_value:.4f}')
            st.write(f'Poder Estadístico Observado: {power_observed:.4f}')
            
            if p_value < alpha_post:
                st.success('El resultado es estadísticamente significativo. Se puede concluir que hay una diferencia entre las versiones A y B.')
            else:
                st.warning('El resultado no es estadísticamente significativo. No se puede concluir que hay una diferencia entre las versiones A y B.')
            
            if power_observed >= 0.8:
                st.success('El poder estadístico es adecuado. El test tiene suficiente sensibilidad para detectar una diferencia.')
            else:
                st.warning('El poder estadístico es bajo. El test puede no tener suficiente sensibilidad para detectar una diferencia.')

            # Graficar las probabilidades de ser mejor de las distribuciones a posteriori
            st.subheader('Distribuciones a Posteriori')
            posterior_A = np.random.beta(conversions_A + 1, visits_A - conversions_A + 1, 10000)
            posterior_B = np.random.beta(conversions_B + 1, visits_B - conversions_B + 1, 10000)
            
            plt.figure(figsize=(10, 5))
            plt.hist(posterior_A, bins=50, alpha=0.5, label='Versión A')
            plt.hist(posterior_B, bins=50, alpha=0.5, label='Versión B')
            plt.axvline(x=np.mean(posterior_A), color='blue', linestyle='--')
            plt.axvline(x=np.mean(posterior_B), color='orange', linestyle='--')
            plt.legend()
            plt.xlabel('Tasa de Conversión')
            plt.ylabel('Frecuencia')
            plt.title('Distribuciones a Posteriori')
            st.pyplot(plt)
            
            prob_A_better_than_B = np.mean(posterior_A > posterior_B)

            prob_B_better_than_A = np.mean(posterior_B > posterior_A)
            st.write(f'Probabilidad de que la Versión A sea mejor que la Versión B: {prob_A_better_than_B:.2f}')
            st.write(f'Probabilidad de que la Versión B sea mejor que la Versión A: {prob_B_better_than_A:.2f}')
    
if __name__ == '__main__':
    main()
