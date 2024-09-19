##from turtle import width
import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import numpy as np
import plotly.graph_objects as go

##pile_punch_dir = (os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
##+ '/pages/pile_punch/')
##sys.path.append(pile_punch_dir)
st.header('Расчет на продавливание колонной фундаментной плиты, опирающейся на сваи')



if 'b' not in ss: ss['b'] = 50.0
if 'h' not in ss: ss['h'] = 100.0
if 'h0' not in ss: ss['h0'] = 125.0
if 'cL' not in ss: ss['cL'] = 60.0
if 'cR' not in ss: ss['cR'] = 60.0
if 'cB' not in ss: ss['cB'] = 70.0
if 'cT' not in ss: ss['cT'] = 70.0
if 'is_cL' not in ss: ss['is_cL'] = True
if 'is_cR' not in ss: ss['is_cR'] = True
if 'is_cB' not in ss: ss['is_cB'] = True
if 'is_cT' not in ss: ss['is_cT'] = True
if 'center' not in ss: ss['center'] = [25.0, 50.0]
if 'centerM' not in ss: ss['centerM'] = [25.0, 50.0]


def generate_piles (b, h, h0, cL, is_cL, cR, is_cR, cB, is_cB, cT, is_cT):
    pile_size = 30
    piles = []
    piles_color = []
    if is_cL:
        pile_x = [-cL, -cL-pile_size, -cL-pile_size, -cL, -cL]
        pile_y = [h/2-pile_size/2, h/2-pile_size/2, h/2+pile_size/2, h/2+pile_size/2, h/2-pile_size/2]
        piles.append([pile_x, pile_y])
        if cL<0.4*h0: piles_color.append('red')
        else: piles_color.append('black')
    if is_cR:
        pile_x = [b+cR, b+cR+pile_size, b+cR+pile_size, b+cR, b+cR]
        pile_y = [h/2-pile_size/2, h/2-pile_size/2, h/2+pile_size/2, h/2+pile_size/2, h/2-pile_size/2]
        piles.append([pile_x, pile_y])
        if cR<0.4*h0: piles_color.append('red')
        else: piles_color.append('black')
    if is_cB:
        pile_x = [b/2-pile_size/2, b/2+pile_size/2, b/2+pile_size/2, b/2-pile_size/2, b/2-pile_size/2]
        pile_y = [-cB, -cB, -cB-pile_size, -cB-pile_size, -cB]
        piles.append([pile_x, pile_y])
        if cB<0.4*h0: piles_color.append('red')
        else: piles_color.append('black')
    if is_cT:
        pile_x = [b/2-pile_size/2, b/2+pile_size/2, b/2+pile_size/2, b/2-pile_size/2, b/2-pile_size/2]
        pile_y = [h+cT, h+cT, h+cT+pile_size, h+cT+pile_size, h+cT]
        piles.append([pile_x, pile_y])
        if cT<0.4*h0: piles_color.append('red')
        else: piles_color.append('black')
    return piles, piles_color

def find_contour_geometry (V, M, Rbt, h0, F, Mx, My, deltaM, xcol, ycol):
    #V - массив координат линий контура [   [[x1,x2], [y1,y2]],    [[x1,x2], [y1,y2]], ...   ]
    #M - вектор масс линий
    #Объявляем нулевыми суммарные длину и моменты инерции
    V = np.array(V)
    M = np.array(M)
    #print(V1)
    Lsum, Sx, Sy, Ix, Iy = 0, 0, 0, 0, 0
    LsumM, SxM, SyM, IxM, IyM = 0, 0, 0, 0, 0
    Fult = 0
    #Присваеваем максимум и минимум координат
    #в соответствии с координатами первой точки первой линии
    xmin0, xmax0 = V[:,0].min(), V[:,0].max()
    ymin0, ymax0 = V[:,1].min(), V[:,1].max()
    j = 0
    for i in V:
        #Извлекаем координаты начала и конца i-го участка
        x1, x2 = i[0]
        y1, y2 = i[1]
        #Вычисляем длину i-го участка
        L_i = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
        #Добавляем его длину в суммарную
        Lsum, LsumM = Lsum + L_i, LsumM + L_i*M[j]
        #Вычисляем координаты центра i-го участка
        center_i = ((x2 - x1)/2 + x1, (y2 - y1)/2 + y1)
        #Вычисляем статические моменты i-го участка
        Sx_i = L_i*center_i[0]
        Sy_i = L_i*center_i[1]
        #Добавляем их в суммарные
        Sx, SxM = Sx + Sx_i, SxM + Sx_i*M[j]
        Sy, SyM = Sy + Sy_i, SyM + Sy_i*M[j]
        j += 1
    #Длина контура и весовая длина контура
    Lsum, LsumM = round(Lsum, 2), round(LsumM,2)
    #Предельная сила, воспринимаемая бетоном
    Fbult = LsumM*Rbt*h0
    #Коэффициент использования по продольной силе
    kF= F/Fbult
    #Вычисляем координаты центра тяжести всего контура
    xc, xcM = Sx/Lsum, SxM/LsumM
    yc, ycM = Sy/Lsum, SyM/LsumM
    ex, exM = xc - xcol, xcM - xcol
    ey, eyM = yc - ycol, ycM - ycol
    #Расчет характеристик "без масс"
    for i in V:
        #Вычисляем координаты минимума и максимума относительно геометрического центра тяжести
        xmin, xmax = xmin0 - xc, xmax0 - xc
        ymin, ymax = ymin0 - yc, ymax0 - yc
        #Извлекаем координаты начала и конца i-го участка
        #и пересчитываем их относительно центра тяжести
        x1, x2 = i[0] - xc
        y1, y2 = i[1] - yc
        #Вычисляем центр тяжести
        x0 = (x2 - x1)/2 + x1
        y0 = (y2 - y1)/2 + y1
        #Вычисляем длину i-го участка
        L_i = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
        #Длина i-го участка вдоль оси x и y
        Lx_i = ((x1 - x2)**2)**0.5
        Ly_i = ((y1 - y2)**2)**0.5
        #Собственные моменты инерции
        Ix0_i = Lx_i**3/12
        Iy0_i = Ly_i**3/12
        #Моменты инерции относительно центра тяжести
        Ix_i = Ix0_i + L_i*x0**2
        Iy_i = Iy0_i + L_i*y0**2
        Ix = Ix + Ix_i
        Iy = Iy + Iy_i
    Wxl, Wxr = Ix/abs(xmin), Ix/xmax
    Wxmin = min(Wxl, Wxr)
    Wyb, Wyt = Iy/abs(ymin), Iy/ymax
    Wymin = min(Wyb, Wyt)
    Mbxult = Wxmin*h0*Rbt*M.min()/100
    Mbyult = Wymin*h0*Rbt*M.min()/100
    Mxexc = -F*ex/100
    Myexc = -F*ey/100
    MxexcM = -F*exM/100
    MyexcM = -F*eyM/100
    Mxloc = abs(Mx + Mxexc)
    Myloc = abs(My + Myexc)
    MxlocM = abs(Mx + MxexcM)
    MylocM = abs(My + MyexcM)
    kM = (Mxloc/Mbxult + Myloc/Mbyult)*deltaM
    kMM = (MxlocM/Mbxult + MylocM/Mbyult)*deltaM
    kM = min(kM, kF/2)
    kMM = min(kMM, kF/2)
    kMmax = max(kM, kMM) 
    k = kF + max(kM, kMM) 
    return {'Lsum': Lsum, 'xc': xc, 'yc': yc, 'xcM': xcM, 'ycM': ycM, 
                                     'ex': ex, 'ey': ey, 'exM': exM, 'eyM': eyM,
                                     'Mxexc': Mxexc, 'Myexc': Myexc, 'MxexcM': MxexcM, 'MyexcM': MyexcM,
                                     'Mxloc': Mxloc, 'Myloc': Myloc, 'MxlocM': MxlocM, 'MylocM': MylocM, 
                                     'Fbult': Fbult, 'Mbxult': Mbxult, 'Mbyult': Mbyult,
                                     'kF': kF, 'kM': kM, 'kMM': kMM, 'kMmax': kMmax , 'k': k}

def generate_black_contours (b, h, h0, cL, is_cL, cR, is_cR, cB, is_cB, cT, is_cT):
    contour = []
    if is_cL:
        contour_x = [-cL, -cL]
        contour_y = [-cB, h+cT]
        contour.append([contour_x, contour_y])
    if is_cR:
        contour_x = [b+cR, b+cR]
        contour_y = [-cB, h+cT]
        contour.append([contour_x, contour_y])
    if is_cB:
        contour_x = [-cL, b+cR]
        contour_y = [-cB, -cB]
        contour.append([contour_x, contour_y])
    if is_cT:
        contour_x = [-cL, b+cR]
        contour_y = [h+cT, h+cT]
        contour.append([contour_x, contour_y])
    return contour

def generate_blue_contours (b, h, h0, cL, is_cL, cR, is_cR, cB, is_cB, cT, is_cT):
    contour = []
    cL0 = round(max(min(cL,h0),0.4*h0),1)
    cR0 = round(max(min(cR,h0),0.4*h0),1)
    cT0 = round(max(min(cT,h0),0.4*h0),1)
    cB0 = round(max(min(cB,h0),0.4*h0),1)
    contour_gamma = []
    contour_colour = []
    contour_sides = []
    contour_len = []
    if is_cL:
        contour_x = [-cL0/2, -cL0/2]
        contour_y = [-cB0/2, h+cT0/2]
        if not is_cT: contour_y[1] = h+cT
        if not is_cB: contour_y[0] = -cB
        contour_gamma.append(round(h0/cL0,2))
        contour.append([contour_x, contour_y])
        contour_sides.append('левый')
        L = ((contour_x[1]-contour_x[0])**2 + (contour_y[1]-contour_y[0])**2)**0.5
        contour_len.append(L)
    if is_cR:
        contour_x = [b+cR0/2, b+cR0/2]
        contour_y = [-cB0/2, h+cT0/2]
        if not is_cT: contour_y[1] = h+cT
        if not is_cB: contour_y[0] = -cB
        contour_gamma.append(round(h0/cR0,2))
        contour.append([contour_x, contour_y])
        contour_sides.append('правый')
        L = ((contour_x[1]-contour_x[0])**2 + (contour_y[1]-contour_y[0])**2)**0.5
        contour_len.append(L)
    if is_cB:
        contour_x = [-cL0/2, b+cR0/2]
        contour_y = [-cB0/2, -cB0/2]
        if not is_cL: contour_x[0] = -cL
        if not is_cR: contour_x[1] = b+cR
        contour_gamma.append(round(h0/cB0,2))
        contour.append([contour_x, contour_y])
        contour_sides.append('нижний')
        L = ((contour_x[1]-contour_x[0])**2 + (contour_y[1]-contour_y[0])**2)**0.5
        contour_len.append(L)
    if is_cT:
        contour_x = [-cL0/2, b+cR0/2]
        contour_y = [h+cT0/2, h+cT0/2]
        if not is_cL: contour_x[0] = -cL
        if not is_cR: contour_x[1] = b+cR
        contour_gamma.append(round(h0/cT0,2))
        contour.append([contour_x, contour_y])
        contour_sides.append('верхний')
        L = ((contour_x[1]-contour_x[0])**2 + (contour_y[1]-contour_y[0])**2)**0.5
        contour_len.append(L)
    return contour, contour_gamma, contour_sides, contour_len

def generate_red_contours (b, h, h0, cL, is_cL, cR, is_cR, cB, is_cB, cT, is_cT):
    pile_size = 30
    contour = []
    if not is_cL:
        contour_x = [-cL, -cL]
        contour_y = [-cB-2*pile_size, h+cT+2*pile_size]
        contour.append([contour_x, contour_y])
    if not is_cR:
        contour_x = [b+cR, b+cR]
        contour_y = [-cB-2*pile_size, h+cT+2*pile_size]
        contour.append([contour_x, contour_y])
    if not is_cB:
        contour_x = [-cL-2*pile_size, b+cR+2*pile_size]
        contour_y = [-cB, -cB]
        contour.append([contour_x, contour_y])
    if not is_cT:
        contour_x = [-cL-2*pile_size, b+cR+2*pile_size]
        contour_y = [h+cT, h+cT]
        contour.append([contour_x, contour_y])
    return contour
    


def draw_scheme(b, h, h0,
                cL, is_cL, cR, is_cR, cB, is_cB, cT, is_cT,
                piles, piles_color,
                black_contours, red_contours, blue_contours, contour_gamma, center, centerM):
    fig = go.Figure()

    #fig.add_trace(go.Scatter(x=[-cL-10, -cL-20, -cL-10],
    #                     y=[-cB-30, -cB-20, -cB-10],
    #                     mode='lines',
    #                     line=dict(color='blue',
    #                              shape='spline',
    #                              width=1#The  width is proportional to the edge weight
    #                             ),
    #                    hoverinfo='none'
    #                   )
    #            )

    fig.add_trace(go.Scatter(x=[center[0]], y=[center[1]], showlegend=False, mode="markers", marker_symbol=4, marker_size=10, line=dict(color='green')))
    fig.add_trace(go.Scatter(x=[centerM[0]], y=[centerM[1]], showlegend=False, mode="markers", marker_symbol=4, marker_size=10, line=dict(color='red')))
    #Добавляем колонну
    xx = [0, b, b, 0, 0]
    yy = [0, 0, h, h, 0]
    fig.add_trace(go.Scatter(x=[0, b], y=[h/2, h/2], showlegend=False, mode='lines', line=dict(color='black', width=0.5)))
    fig.add_trace(go.Scatter(x=[b/2, b/2], y=[0, h], showlegend=False, mode='lines', line=dict(color='black', width=0.5)))
    fig.add_trace(go.Scatter(x=xx, y=yy, showlegend=False, mode='lines', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=xx, y=yy, showlegend=False, fill='toself', mode='lines', line=dict(color='black'), fillpattern=dict(fgcolor='black', size=10, fillmode='replace', shape="/")))
    arrows_props = dict(arrowcolor="black",arrowsize=3,arrowwidth=0.5,arrowhead=3)
    text_props = dict(font=dict(color='black',size=14), showarrow=False, bgcolor="#ffffff")
    #Добавляем сваи
    for i in range(len(piles)):
        pile = piles[i]
        col = piles_color[i]
        fig.add_trace(go.Scatter(x=pile[0], y=pile[1], showlegend=False, fill='toself', mode='lines', line=dict(color=col), fillpattern=dict(fgcolor=col, size=10, fillmode='replace', shape="/")))
    for contour in red_contours:
        fig.add_trace(go.Scatter(x=contour[0], y=contour[1], showlegend=False, mode='lines', line=dict(color='red')))
    for contour in black_contours:
        fig.add_trace(go.Scatter(x=contour[0], y=contour[1], showlegend=False, mode='lines', line=dict(color='black')))
    for contour in blue_contours:
        fig.add_trace(go.Scatter(x=contour[0], y=contour[1], showlegend=False, mode='lines', line=dict(color='blue')))
    for i in range(len(contour_gamma)):
        cont =  blue_contours[i]
        cx = cont[0][0] + (cont[0][1]-cont[0][0])/2
        cy = cont[1][0] + (cont[1][1]-cont[1][0])/2
        lx = (cont[0][1]-cont[0][0])
        ly = (cont[1][1]-cont[1][0])
        l = (lx**2+ly**2)**0.5
        if cont[1][0] == cont[1][1]:
            ang = 0
            yan = 'bottom'
            yan2 = 'top'
            if cy<=0:
                yan = 'top'
                yan2 = 'bottom'
            xan = 'center'
            xan2 = 'center'
            cx = b/2
        else:
            ang = 270
            yan = 'middle'
            yan2 = 'middle'
            xan = 'right'
            xan2 = 'left'
            if cx>=0:
                xan = 'left'
                xan2 = 'right'
            cy = h/2
        fig.add_annotation(dict(x=cx, y=cy, text=f'{float(contour_gamma[i]):g}', textangle=ang, yanchor=yan,xanchor=xan, **text_props))
        fig.add_annotation(dict(x=cx, y=cy, text=f'{float(l):g}', textangle=ang, yanchor=yan2,xanchor=xan2, **text_props))
        if cx<=0.0 and lx == 0.0:
            fig.add_annotation(dict(x=cx/2, y=0, text=f'{float(abs(cx)):g}', textangle=0, yanchor='bottom',xanchor='center', **text_props))
            fig.add_annotation(x=cx, y=0, ax=-0.9*cx, ay=0, **arrows_props)
            fig.add_annotation(x=0, y=0, ax=0.9*cx, ay=0, **arrows_props)
        if cx>=0.0 and lx == 0.0:
            fig.add_annotation(dict(x=b+(cx-b)/2, y=0, text=f'{float(abs(cx-b)):g}', textangle=0, yanchor='bottom',xanchor='center', **text_props))
            fig.add_annotation(x=b, y=0, ax=0.9*(cx-b), ay=0, **arrows_props)
            fig.add_annotation(x=cx, y=0, ax=-0.9*(cx-b), ay=0, **arrows_props)
        if cy<=0.0 and ly == 0.0:
            fig.add_annotation(dict(x=b, y=cy/2, text=f'{float(abs(cy)):g}', textangle=270, yanchor='middle',xanchor='right', **text_props))
            fig.add_annotation(x=b, y=0, ax=0, ay=-0.9*cy, **arrows_props)
            fig.add_annotation(x=b, y=cy, ax=0, ay=0.9*cy, **arrows_props)
        if cy>=0.0 and ly == 0.0:
            fig.add_annotation(dict(x=b, y=h+(cy-h)/2, text=f'{float(abs(cy-h)):g}', textangle=270, yanchor='middle',xanchor='right', **text_props))
            fig.add_annotation(x=b, y=h, ax=0, ay=-0.9*(cy-h), **arrows_props)
            fig.add_annotation(x=b, y=cy, ax=0, ay=0.9*(cy-h), **arrows_props)
    #Ширина колонны
    fig.add_annotation(dict(x=b/2, y=0, text=f'{float(b):g}', textangle=0, yanchor='bottom',xanchor='center', **text_props))
    #Расстояние до левой грани
    if cL>0:
        fig.add_annotation(x=-cL, y=h, ax=0.9*cL, ay=0, **arrows_props)
        fig.add_annotation(x=0, y=h, ax=-0.9*cL, ay=0, **arrows_props)
        fig.add_annotation(dict(x=-cL/2, y=h, text=f'{float(cL):g}', textangle=0, yanchor='bottom',xanchor='center', **text_props))
    if not is_cL:
        if cL>0:
            fig.add_annotation(x=-cL, y=0, ax=0.9*cL, ay=0, **arrows_props)
            fig.add_annotation(x=0, y=0, ax=-0.9*cL, ay=0, **arrows_props)
            fig.add_annotation(dict(x=-cL/2, y=0, text=f'{float(cL):g}', textangle=0, yanchor='bottom',xanchor='center', **text_props))
    #Расстояние до правой грани
    if cR>0:
        fig.add_annotation(x=b, y=h, ax=0.9*cR, ay=0, **arrows_props)
        fig.add_annotation(x=b+cR, y=h, ax=-0.9*cR, ay=0, **arrows_props)
        fig.add_annotation(dict(x=b+cR/2, y=h, text=f'{float(cR):g}', textangle=0, yanchor='bottom',xanchor='center', **text_props))
    if not is_cR:
        if cR>0:
            fig.add_annotation(x=b, y=0, ax=0.9*cR, ay=0, **arrows_props)
            fig.add_annotation(x=b+cR, y=0, ax=-0.9*cR, ay=0, **arrows_props)
            fig.add_annotation(dict(x=b+cR/2, y=0, text=f'{float(cR):g}', textangle=0, yanchor='bottom',xanchor='center', **text_props))
    #Высота колонны
    fig.add_annotation(dict(x=b, y=h/2, text=f'{float(h):g}', textangle=270, yanchor='middle',xanchor='right', **text_props))
    #Расстояние до нижней грани
    if cB>0:
        fig.add_annotation(x=0, y=0, ax=0, ay=0.9*cB, **arrows_props)
        fig.add_annotation(x=0, y=-cB, ax=0, ay=-0.9*cB, **arrows_props)
        fig.add_annotation(dict(x=0, y=-cB/2, text=f'{float(cB):g}', textangle=270, yanchor='middle',xanchor='right', **text_props))
    if not is_cB:
        if cB>0:
            fig.add_annotation(x=b, y=0, ax=0, ay=0.9*cB, **arrows_props)
            fig.add_annotation(x=b, y=-cB, ax=0, ay=-0.9*cB, **arrows_props)
            fig.add_annotation(dict(x=b, y=-cB/2, text=f'{float(cB):g}', textangle=270, yanchor='middle',xanchor='right', **text_props))
    #Расстояние до верхней грани
    if cT>0:
        fig.add_annotation(x=0, y=h, ax=0, ay=-0.9*cT, **arrows_props)
        fig.add_annotation(x=0, y=h+cT, ax=0, ay=0.9*cT, **arrows_props)
        fig.add_annotation(dict(x=0, y=h+cT/2, text=f'{float(cT):g}', textangle=270, yanchor='middle',xanchor='right', **text_props))
    if not is_cT:
        if cT>0:
            fig.add_annotation(x=b, y=h, ax=0, ay=-0.9*cT, **arrows_props)
            fig.add_annotation(x=b, y=h+cT, ax=0, ay=0.9*cT, **arrows_props)
            fig.add_annotation(dict(x=b, y=h+cT/2, text=f'{float(cT):g}', textangle=270, yanchor='middle',xanchor='right', **text_props))
    fig.update_yaxes(scaleanchor="x",scaleratio=1,title="y")
    fig.update_xaxes(dict(title="x", visible=False))
    fig.update_yaxes(visible=False)
    fig.update_layout(autosize=True, margin={'l': 0, 'r': 0, 't': 0, 'b': 0})
    return fig


cols_size = [1 for i in range(8)]
cols = st.columns(cols_size)
Rbt = cols[0].number_input(label='$R_{bt}$, МПа', step=0.05, format="%.2f", value=1.4, min_value=0.1, max_value=5.0, label_visibility="visible")
Rbt = 0.01019716213*Rbt
h0 = cols[1].number_input(label='$h_0$, см', key='h0', step=5.0, format="%.1f", value=125.0, min_value=1.0, max_value=500.0, label_visibility="visible")
b = cols[2].number_input(label='$b$, см', key='b', step=5.0, format="%.2f", value=50.0, min_value=1.0, max_value=500.0, label_visibility="visible")
h = cols[3].number_input(label='$h$, см', key='h', step=5.0, format="%.2f", value=100.0, min_value=1.0, max_value=500.0, label_visibility="visible")
F = cols[4].number_input(label='$F$, тс', step=0.5, format="%.1f", value=1400.0, min_value=1.0, max_value=50000.0, label_visibility="visible")
Mx = cols[5].number_input(label='$M_x$, тсм', step=0.5, format="%.1f", value=90.0, label_visibility="visible")
My = cols[6].number_input(label='$M_y$, тсм', step=0.5, format="%.1f", value=120.0, label_visibility="visible")
deltaM = cols[7].number_input(label='$\delta_M$', step=0.1, format="%.2f", value=0.5, min_value=0.0, max_value=2.0, label_visibility="visible")

black_contours = generate_black_contours(ss['b'], ss['h'], ss['h0'], ss['cL'], ss['is_cL'], ss['cR'], ss['is_cR'], ss['cB'], ss['is_cB'], ss['cT'], ss['is_cT'])
red_contours = generate_red_contours(ss['b'], ss['h'], ss['h0'], ss['cL'], ss['is_cL'], ss['cR'], ss['is_cR'], ss['cB'], ss['is_cB'], ss['cT'], ss['is_cT'])
blue_contours, contour_gamma, contour_sides, contour_len = generate_blue_contours(ss['b'], ss['h'], ss['h0'], ss['cL'], ss['is_cL'], ss['cR'], ss['is_cR'], ss['cB'], ss['is_cB'], ss['cT'], ss['is_cT'])
piles, piles_color = generate_piles(ss['b'], ss['h'], ss['h0'], ss['cL'], ss['is_cL'], ss['cR'], ss['is_cR'], ss['cB'], ss['is_cB'], ss['cT'], ss['is_cT'])
num_elem = len(blue_contours)
rez = 0
if num_elem>=2:
    rez = find_contour_geometry(blue_contours, contour_gamma, Rbt, h0, F, Mx, My, deltaM, b/2, h/2)
    ss['center'] = [rez['xc'], rez['yc']]
    ss['centerM'] = [rez['xcM'], rez['ycM']]
    #st.write(rez)



fig = draw_scheme(ss['b'], ss['h'], ss['h0'],
                  ss['cL'], ss['is_cL'], ss['cR'], ss['is_cR'], ss['cB'], ss['is_cB'], ss['cT'], ss['is_cT'],
                  piles, piles_color, black_contours, red_contours, blue_contours, contour_gamma, ss['center'], ss['centerM'])





with st.expander('Описание исходных данных'):
    st.write(''' $b$ и $h$ - ширина и высота поперечного сечения сечения колонны, см; ''')
    st.write(''' $h_0$ - рабочая высота поперечного сечения фундаментной плиты, см; ''')
    st.write(''' $c_L$ и $c_R$ - расстояние в свету до левой и правой сваи (грани плиты) от грани колонны, см; ''')
    st.write(''' $c_B$ и $c_T$ - расстояние в свету до нижней и верхней сваи (грани плиты) от грани колонны, см; ''')
    st.write(''' $R_{bt}$ - расчетное сопротивление на растяжение материала фундаментной плиты, МПа; ''')
    st.write(''' $F$ - продавливающее усилие, тс; ''')
    st.write(''' $M_x$ - сосредоточенные момент в ПЛОСКОСТИ оси $x$ (относительно оси $y$), тсм; ''')
    st.write(''' $M_y$ - сосредоточенные момент в ПЛОСКОСТИ оси $y$ (относительно оси $x$), тсм; ''')
    st.write(''' $\delta_M$ - понижающий коэффициент к сосредоточенным моментам. ''')



cols = st.columns([0.2, 1, 0.25], vertical_alignment="center")
cols2 = cols[1].columns([1, 1, 1], vertical_alignment="center")
##cols2[0].write('$c_T$, см')
cols2[1].number_input(label='$c_T$, см', key='cT', step=5.0, format="%.1f", min_value=0.0, max_value=500.0)
cols2[2].toggle('Контур_сверху', key='is_cT', label_visibility="collapsed")

cols = st.columns([0.2, 1, 0.2], vertical_alignment="center")
cols[1].plotly_chart(fig, use_container_width=True)
cols[0].write('$c_L$, см')
cols[0].number_input(label='$cR1$, см', key='cL', step=5.0, format="%.1f", min_value=0.0, max_value=500.0, label_visibility="collapsed")
cols[0].toggle('Контур_слева', key='is_cL', label_visibility="collapsed")
cols[2].write('$c_R$, см')
cols[2].number_input(label='$b1$, см', key='cR', step=5.0, format="%.1f", min_value=0.0, max_value=500.0, label_visibility="collapsed")
cols[2].toggle('Контур_справа', key='is_cR', label_visibility="collapsed")
cols2 = cols[1].columns([1, 1, 1], vertical_alignment="center")
##cols2[0].write('$c_B$, см')
cols2[1].number_input(label='$c_B$, см', key='cB', step=5.0, format="%.1f", min_value=0.0, max_value=500.0)
cols2[2].toggle('Контур_снизу', key='is_cB', label_visibility="collapsed")

if num_elem<2:
    st.write('В расчете должно быть минимум два участка!')
    st.stop()



with st.expander('Расчетные выкладки'):

    st.write(':red[ТЕКСТОВОЕ ОПИСАНИЕ НЕАКТУАЛЬНО, КОРРЕКТИРУЕТСЯ. КОЭФФИЦИЕНТЫ ИСПОЛЬЗОВАНИЯ АКТУАЛЬНЫ]')
    st.write(f'Проверка (корректировка) значений $c_i$ из условия $0.4 \cdot h_0 = {float(0.4*h0):g} \le c_i \le h_0 = {float(h0):g}$.')
    cL, cR, cB, cT = 0, 0, 0, 0
    if ss['is_cL']:
        cL = round(max(min(ss['cL'],h0),0.4*h0),1)
    else: cL = round(ss['cL'],1)

    if ss['is_cR']:
        cR = round(max(min(ss['cR'],h0),0.4*h0),1)
    else: cR = round(ss['cR'],1)

    if ss['is_cB']:
        cB = round(max(min(ss['cB'],h0),0.4*h0),1)
    else: cB = round(ss['cB'],1)

    if ss['is_cT']:
        cT = round(max(min(ss['cT'],h0),0.4*h0),1)
    else: cT = round(ss['cT'],1)

    st.write(cL, cR, cB, cT)

    string = f'В расчете принимаем: '

    if ss['is_cL']: string += '$c_L = ' + str({cL}) + '$см; '
    if ss['is_cR']: string += '$c_R = ' + str({cR}) + '$см; '
    if ss['is_cB']: string += '$c_B = ' + str({cB}) + '$см; '
    if ss['is_cT']: string += '$c_T = ' + str({cT}) + '$см; '
    st.write(string)

    st.write('Вычисляем повышающие коэффициенты к прочности бетона $1.0 \le \\gamma_i = h_0/c_i \le 2.5$.')
    gammaL, gammaR, gammaB, gammaT = 1, 1, 1, 1
    if ss['is_cL']: gammaL = round(max(min(h0/cL,2.5),0.4),2)
    if ss['is_cR']: gammaR = round(max(min(h0/cR,2.5),0.4),2)
    if ss['is_cB']: gammaB = round(max(min(h0/cB,2.5),0.4),2)
    if ss['is_cT']: gammaT = round(max(min(h0/cT,2.5),0.4),2)

    string = f'В расчете принимаем: '
    if ss['is_cL']: string += '$\\gamma_L = ' + str({gammaL}) + '$; '
    if ss['is_cR']: string += '$\\gamma_R = ' + str({gammaR}) + '$; '
    if ss['is_cB']: string += '$\\gamma_B = ' + str({gammaB}) + '$; '
    if ss['is_cT']: string += '$\\gamma_T = ' + str({gammaT}) + '$; '
    st.write(string)

    LL = 0
    LR = 0
    LT = 0
    LB = 0
    string = 'Длины участков контура составляют: '
    if ss['is_cL']:
        LL = LL + h
        if ss['is_cB']: LL = LL + cB/2
        if ss['is_cT']: LL = LL + cT/2
        string += f'левый, $L_L= {LL}$ см; '

    if ss['is_cR']:
        LR = LR + h
        if ss['is_cB']: LR = LR + cB/2
        if ss['is_cT']: LR = LR + cT/2
        string += f'правый, $L_R= {LR}$ см; '

    if ss['is_cB']:
        LB = LB + b
        if ss['is_cL']: LB = LB + cL/2
        if ss['is_cR']: LB = LB + cL/2
        string += f'нижний, $L_B= {LB}$ см; '
    
        
    st.write(string)
    st.write(LL)
    

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


st.write('Коэффициент использования по продольной силе $k_F=' + str(round(rez['kF'],3)) + '$.')
st.write('Коэффициент использования по моментам $k_М=' + str(round(rez['kMmax'],3)) + '$.')
st.write('Суммарный коэффициент использования $k=' + str(round(rez['k'],3)) + '$.')





