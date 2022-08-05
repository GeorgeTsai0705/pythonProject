import os
import time
import pyautogui

def get_data():
    with open("Result.txt",'r') as f:
        Wavelength = []
        T = []

        for line in f.readlines():
            if line[0] !=" ":
                continue
            Wavelength.append(float(line[5:13]))
            T.append(float(line[24:32]))
    return Wavelength, T

def rank_result(T):

    #### RED FILTER CAL
    Transmission = sum(T[301:401])/len(T[301:401])
    #print("Transmission", Transmission)
    Suppression = sum(T[101:301])/len(T[101:301])
    #print("Suppression", Suppression)
    Purity = Transmission - Suppression
    #print("Purity", Purity)
    Red_Score = Transmission + Purity + (1 - Suppression)
    #print("Red_Score", Red_Score)

    #### GREEN FILTER CAL
    Transmission = sum(T[201:301])/len(T[201:301])
    Suppression = (sum(T[101:201])/len(T[101:201]) + sum(T[301:401])/len(T[301:401]))/2
    Purity = Transmission - Suppression
    Green_Score = Transmission + Purity + (1 - Suppression)

    #### BLUE FILTER CAL
    Transmission = sum(T[101:201])/len(T[101:201])
    Suppression = sum(T[201:401])/len(T[201:401])
    Purity = Transmission - Suppression
    Blue_Score = Transmission + Purity + (1 - Suppression)

    return [Transmission, Suppression, Purity, Red_Score, Green_Score,Blue_Score ]

def mouse_control():
    Thickness_Position = [(200,195), (200,224), (200,254), (200,267), (200,290), (200,311),
                          (200,333), (200,355), (200, 377), (200, 400), (200, 422), (200, 443)]

    for i in range(7):
        i_button = True

        for j in range(7):
            Num1 = (i+1)*5+15
            Num2 = (j+1)*5+15
            Cont = 0

            for ele in Thickness_Position:

                # if we already change Num1, save time skip the process of keying number
                if i_button and Cont % 2 ==0:

                    pyautogui.click(x=ele[0], y=ele[1], clicks=2)
                    time.sleep(1)

                    if not pyautogui.locateCenterOnScreen("Button/OK.png"):
                        pyautogui.click(x=ele[0], y=ele[1], clicks=2)

                    pyautogui.press('tab')
                    pyautogui.typewrite(str(Num1))

                elif Cont % 2 ==0:
                    pass

                else:

                    pyautogui.click(x=ele[0], y=ele[1], clicks=2)
                    time.sleep(1)

                    if not pyautogui.locateCenterOnScreen("Button/OK.png"):
                        pyautogui.click(x=ele[0], y=ele[1], clicks=2)

                    pyautogui.press('tab')
                    pyautogui.typewrite(str(Num2))

                Cont += 1
                time.sleep(1)
                pyautogui.click((pyautogui.locateCenterOnScreen("Button/OK.png")))

            pyautogui.hotkey('alt', 't')
            pyautogui.press(['tab','right','right'])
            time.sleep(1)
            pyautogui.click((pyautogui.locateCenterOnScreen("Button/OK3.png")))
            time.sleep(0.5)
            pyautogui.click(x=203, y=43)
            #pyautogui.click((pyautogui.locateCenterOnScreen("Button/Analyse.png")))
            time.sleep(1)
            pyautogui.click((pyautogui.locateCenterOnScreen("Button/Export.png")))
            time.sleep(2)
            pyautogui.click((pyautogui.locateCenterOnScreen("Button/Result_txt.png")))
            time.sleep(1)
            pyautogui.click((pyautogui.locateCenterOnScreen("Button/Save.png")))
            time.sleep(1)
            pyautogui.click((pyautogui.locateCenterOnScreen("Button/YES.png")))
            Wavelength, T = get_data()
            Filter_Result = rank_result(T)

            i_button = False

            with open("Red_Cal.txt", 'a') as f2:
                f2.write(f"{Num1}\t{Num2}\t{Filter_Result[0]:8.2f}\t{Filter_Result[1]:8.2f}\t{Filter_Result[2]:8.2f}\t{Filter_Result[3]:8.2f}\t{Filter_Result[4]:8.2f}\t{Filter_Result[5]:8.2f}\n")

def show_postion():
    while 1:
        time.sleep(0.5)
        P = pyautogui.position()
        print(P)
        S = pyautogui.locateCenterOnScreen("Button/Result_txt.png")
        print(S)

def main():

    time.sleep(2)
    mouse_control()
    #show_postion()

    """
    Wavelength, T = get_data()
    Filter_Result = rank_result(T)
    print(Filter_Result)
    Transmission = sum(T[201:301])/len(T[201:301])
    Suppression = (sum(T[101:201])/len(T[101:201]) + sum(T[301:401])/len(T[301:401]))/2
    Purity = Transmission - Suppression
    Green_Score = Transmission + Purity + (1 - Suppression)
    print(Transmission, Suppression, Purity, Green_Score)
    """


if __name__ == '__main__':
    main()