import streamlit as st
import pandas as pd
from PIL import Image

##pile_punch_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
##+ '/pages/pile_punch/')
##sys.path.append(pile_punch_dir)
image = Image.open("model.png")

st.header('Расчет на продавливание колонной фундаментной плиты, опирающейся на сваи')

with st.expander('Описание исходных данных'):
    st.write(''' $b$ и $h$ - ширина и высота поперечного сечения сечения колонны, см; ''')
    st.write(''' $c_L$ и $c_R$ - расстояние в свету до левой и правой сваи от грани колонны, см; ''')
    st.write(''' $c_B$ и $c_T$ - расстояние в свету до нижней и верхней сваи от грани колонны, см; ''')
    st.write(''' $h_0$ - рабочая высота поперечного сечения фундаментной плиты, см; ''')
    st.write(''' $R_{bt}$ - расчетное сопротивление на растяжение материала фундаментной плиты, МПа; ''')
    st.write(''' $F$ - продавливающее усилие, тс; ''')
    st.write(''' $M_x$ - сосредоточенные момент в ПЛОСКОСТИ оси $x$ (относительно оси $y$), тсм; ''')
    st.write(''' $M_y$ - сосредоточенные момент в ПЛОСКОСТИ оси $y$ (относительно оси $x$), тсм; ''')
    st.write(''' $\delta_M$ - понижающий коэффициент к сосредоточенным моментам. ''')


cols = st.columns([1, 0.35])
cols[0].image(image)

cols2_size = [0.8, 1]
cols2 = cols[1].columns(cols2_size)
cols2[0].write('$R_{bt}$, МПа')
Rbt = cols2[1].number_input(label='$R_bt$, МПа', step=0.05, format="%.2f", value=1.4, min_value=0.1, max_value=5.0, label_visibility="collapsed")
Rbt = 0.01019716213*Rbt

cols2 = cols[1].columns(cols2_size)
cols2[0].write('$h_0$, см')
h0 = cols2[1].number_input(label='$h_0$, см', step=0.5, format="%.2f", value=125.0, min_value=1.0, max_value=500.0, label_visibility="collapsed")

cols2 = cols[1].columns(cols2_size)
cols2[0].write('$b$, см')
b = cols2[1].number_input(label='$b$, см', step=1.0, format="%.2f", value=50.0, min_value=1.0, max_value=500.0, label_visibility="collapsed")

cols2 = cols[1].columns(cols2_size)
cols2[0].write('$h$, см')
h = cols2[1].number_input(label='$h$, см', step=1.0, format="%.2f", value=100.0, min_value=1.0, max_value=500.0, label_visibility="collapsed")

cols2 = cols[1].columns(cols2_size)
cols2[0].write('$c_L$, см')
cL = cols2[1].number_input(label='$c_L$, см', step=0.5, format="%.2f", value=60.0, min_value=1.0, max_value=500.0, label_visibility="collapsed")

cols2 = cols[1].columns(cols2_size)
cols2[0].write('$c_R$, см')
cR = cols2[1].number_input(label='$c_R$, см', step=0.5, format="%.2f", value=60.0, min_value=1.0, max_value=500.0, label_visibility="collapsed")

cols2 = cols[1].columns(cols2_size)
cols2[0].write('$c_B$, см')
cB = cols2[1].number_input(label='$c_B$, см', step=0.5, format="%.2f", value=70.0, min_value=1.0, max_value=500.0, label_visibility="collapsed")

cols2 = cols[1].columns(cols2_size)
cols2[0].write('$c_T$, см')
cT = cols2[1].number_input(label='$c_T$, см', step=0.5, format="%.2f", value=70.0, min_value=1.0, max_value=500.0, label_visibility="collapsed")

cols2 = cols[1].columns(cols2_size)
cols2[0].write('$F$, тс')
F = cols2[1].number_input(label='$F$, тс', step=0.5, format="%.2f", value=1400.0, min_value=1.0, max_value=50000.0, label_visibility="collapsed")

cols2 = cols[1].columns(cols2_size)
cols2[0].write('$M_x$, тсм')
Mx = cols2[1].number_input(label='$M_x$, тсм', step=0.5, format="%.2f", value=90.0, min_value=1.0, max_value=50000.0, label_visibility="collapsed")

cols2 = cols[1].columns(cols2_size)
cols2[0].write('$M_y$, тсм')
My = cols2[1].number_input(label='$M_y$, тсм', step=0.5, format="%.2f", value=120.0, min_value=1.0, max_value=50000.0, label_visibility="collapsed")

cols2 = cols[1].columns(cols2_size)
cols2[0].write('$\delta_M$')
deltaM = cols2[1].number_input(label='$\delta_M$', step=0.1, format="%.2f", value=0.5, min_value=0.0, max_value=2.0, label_visibility="collapsed")

with st.expander('Расчетные выкладки'):
    st.write('Проверка (при необходимости корректировка) значений $c_i$ из условия $0.4 \cdot h_0 \le c_i \le h_0$.')
    cL = round(max(min(cL,h0),0.4*h0),1)
    cR = round(max(min(cR,h0),0.4*h0),1)
    cB = round(max(min(cB,h0),0.4*h0),1)
    cT = round(max(min(cT,h0),0.4*h0),1)
    st.write(f'В расчете принимаем: $c_L = {cL}$см; $c_R = {cR}$см; $c_B = {cB}$см; $c_T = {cT}$см.')

    st.write('Вычисляем повышающие коэффициенты к прочности бетона $1.0 \le \\gamma_i = h_0/c_i \le 2.5$.')
    gammaL = round(max(min(h0/cL,2.5),0.4),2)
    gammaR = round(max(min(h0/cR,2.5),0.4),2)
    gammaB = round(max(min(h0/cB,2.5),0.4),2)
    gammaT = round(max(min(h0/cT,2.5),0.4),2)

    st.write(f'В расчете принимаем: $\\gamma_L = {gammaL}$; $\\gamma_R = {gammaR}$; $\\gamma_B = {gammaB}$; $\\gamma_T = {gammaT}$.')

    st.write('Предельную продавливаюшую силу, воспринимаемую бетоном, вычисляем по формуле:')

    st.write('''$
    F_{b,ult} = R_{bt} \cdot h_0 \cdot \\left[
    (\\gamma_L + \\gamma_R) (h + c_B/2 + c_T/2 ) +
    (\\gamma_B + \gamma_T) (b + c_R/2 + c_L/2 )
    \\right].
    $''')

    Fbult = Rbt*h0*( gammaL*(h+cB/2+cT/2) + gammaR*(h+cB/2+cT/2) + gammaB*(b+cL/2+cR/2) + gammaT*(b+cL/2+cR/2) )
    Fbult = round(Fbult)

    st.write('В результате подстановки значений найдем $F_{b,ult}='+ str(Fbult) + '$тс. $F_{b,ult}/1.5=' + str(round(Fbult/1.5)) + '$тс.' )

    st.write('''НА ДАННЫЙ МОМЕНТ РАСЧЕТ ПРЕДПОЛАГАЕТ СИММЕТРИЧНОЕ РАСПОЛОЖЕНИЕ СВАЙ ВОКРУГ КОЛОННЫ
    (Т.Е. $c_L=c_R$ и $c_B=c_T$) И НЕ УЧИТЫВАЕТ НЕРАВНОМЕРНОСТЬ ПРОЧНОСТНЫХ ХАРАКТЕРИСТИК.
    ПРИ РАСЧЕТЕ ПРЕДЕЛЬНОГО МОМЕНТА ПОВЫШАЮЩИЙ КОЭФФИЦИЕНТ К ПРОЧНОСТИ БЕТОНА ПРИНИМАЕТСЯ МИНИМАЛЬНЫМ ИЗ НАЙДЕННЫХ РАНЕЕ''')
    st.write(f'Моменты инерции расчетного контура и моменты сопротивления в ПЛОСКОСТИ осей $x$ и $y$ вычисляются по формулам:')

    st.write('''$
    I_{bx} = 2 \cdot \\left[ \dfrac{(c_L/2+b+c_R/2)^3}{12} + (c_B/2+h+c_T/2) \cdot \\left( \dfrac{c_L/2 + b + c_R/2}{2} \\right)^2 \\right];
    $''')

    st.write('''$
    I_{by} = 2 \cdot \\left[ \dfrac{(c_B/2+h+c_T/2)^3}{12} + (c_L/2+b+c_R/2) \cdot \\left( \dfrac{c_B/2 + h + c_T/2}{2} \\right)^2 \\right];
    $''')

    st.write('''$
    W_{bx} = \dfrac{I_{bx}}{0.5 \cdot (c_L/2 + b + c_R/2)};
    W_{by} = \dfrac{I_{by}}{0.5 \cdot (c_B/2 + h + c_T/2)}.
    $''')

    Ibx = 2* ( (cL/2+b+cR/2)**3/12 + (cB/2+h+cT/2)*((cL/2 + b + cR/2)/2)**2 )
    Ibx = round(Ibx)
    Wbx = Ibx/(0.5*(cL/2+b+cR/2))
    Wbx = round(Wbx)

    Iby = 2* ( (cB/2+h+cT/2)**3/12 + (cL/2+b+cR/2)*((cB/2 + h + cT/2)/2)**2 )
    Iby = round(Iby)
    Wby = Iby/(0.5*(cB/2+h+cT/2))
    Wby = round(Wby)

    st.write('В результате расчета найдем следующие значения геометрических характеристик расчетного контура:'+
             ' $I_{bx}=' + str(Ibx) + '$см$^3$;'+
             ' $I_{by}=' + str(Iby) + '$см$^3$;'+
             ' $W_{bx}=' + str(Wbx) + '$см$^2$;'+
             ' $W_{by}=' + str(Wby) + '$см$^2$.'
             )

    st.write('Предельные моменты, воспринимаемые расчетным контуром в плоскости осей $x$ и $y$ вычисляются по формулам:')
    st.write('''$M_{bx,ult}=\\gamma \cdot R_{bt} \cdot h_0 \cdot W_{bx}$;
     $M_{by,ult}=\\gamma \cdot R_{bt} \cdot h_0 \cdot W_{by}$.''')

    gamma_min = min(gammaL, gammaR, gammaB, gammaT)
    st.write('''Здесь $\\gamma='''+ str(gamma_min) + '''$ - минимальное значение повышающего коэффициента к прочности бетона, из найденных ранее.''')

    
    Mbxult = Wbx*gamma_min*h0*Rbt/100
    Mbxult=round(Mbxult,1)
    Mbyult = Wby*gamma_min*h0*Rbt/100
    Mbyult=round(Mbyult,1)
    st.write('В результате расчета найдем: $M_{bx,ult}=' + str(Mbxult) + '$тсм; $M_{by,ult}=' + str(Mbyult) + '$тсм.')

    st.write('Проверка прочности выполняется из условия:')

    st.write('$\dfrac{F}{F_{b,ult}} + \\left(\dfrac{\delta_M \cdot Mx}{M_{bx,ult}} + \dfrac{\delta_M \cdot My}{M_{by,ult}}\\right) = k_F + k_M \le 1$.')
    st.write('Здесь $k_F$ и $k_M$ - коэффициенты использования сечения по силе и моментам соответственно, причем $k_M \le 0.5\cdot k_F$.')

kF = F/Fbult
kF = round(kF,3)
kM = deltaM*(Mx/Mbxult + My/Mbyult)
kM = min(kM, kF*0.5)
kM = round(kM,3)
k = kM + kF
st.write('Коэффициент использования по продольной силе $k_F=' + str(kF) + '$.')
st.write('Коэффициент использования по моментам $k_М=' + str(kM) + '$.')
st.write('Суммарный коэффициент использования $k=' + str(k) + '$.')





