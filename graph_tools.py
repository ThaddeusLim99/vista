import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor

from dataclasses import dataclass
import matplotlib
from matplotlib.backend_bases import MouseButton

import re

#gets the substring from the end of a string till the first forward
#or backwards slash encountered
def get_folder(string: str):
    match = re.search(r"([\\/])([^\\/]+)$", string)
    if match:
        result = match.group(2)
        return result
    return None

class InteractiveDataRateGraph:
    def __init__(self,xBarData,yBarData,yBarAverageData,windowTitle,graphTitle,xlabel,ylabel,isSimple,vistaoutput_path,numScenes):
        self.fig1, self.ax1 = plt.subplots()
        self.fig2, self.ax2 = plt.subplots()
        self.fig3, self.ax3 = plt.subplots()
        self.fig4, self.ax4 = plt.subplots()
        self.dcAxes = []
        self.annotations = {}
        self.xBarData = xBarData
        self.yBarData = yBarData
        self.yBarAverageData = yBarAverageData
        self.windowTitle = windowTitle
        self.graphTitle = graphTitle
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.isSimple = isSimple
        self.vistaoutput_path = vistaoutput_path
        self.numScenes = numScenes
        self.setup_graph()
        self.connect_graph()

    def getGraph(self):
        if self.numScenes > 1: 
            return self.fig1, self.ax1, self.fig2, self.ax2, self.fig3, self.ax3, self.fig4, self.ax4      
        else:
            return self.fig1, self.ax1, self.fig3, self.ax3   
    
    def setup_graph(self):
        # create plots, saving axes/line/annotation for lookup
        @dataclass
        class DataClassAxes:
            fig: str
            ax: plt.Axes
            lines: list[matplotlib.lines.Line2D]
            annotations: list[matplotlib.text.Annotation]
            desc: list[str] #O:Original,A:Average

        if self.isSimple:
            colourScheme = [['g','m'],['b','y']]
        else:
            colourScheme = [['r','c'],['b','y']]

        #number 2 (show original outputs for multiple graphs with only 1 y-scale)
        #fig1, ax1 = plt.subplots()
        self.fig1.canvas.manager.set_window_title(f'{self.windowTitle}') 
        self.fig1.suptitle(f"{self.graphTitle} with only 1 y-scale", fontsize=12)
        self.ax1.set_ylabel(f"{self.ylabel} {get_folder(self.vistaoutput_path[0])}",color='r')
        self.ax1.tick_params(axis='y', colors='r') 
        
        self.ax1.set_xlabel(f"{self.xlabel}")
        self.fig1.tight_layout()
        self.fig1.set_label('fig1')
        
        for i in range(self.numScenes):
            #ORIGINAL PLOT
            line = []
            line.append(self.ax1.plot(self.xBarData[i][:, 0], self.yBarData[i][:, 0],\
                f'{colourScheme[np.mod(i,2)][0]}', label=f'Original: {get_folder(self.vistaoutput_path[i])}'))

            self.fig1.legend()

            annotations = []
            self.dcAxes.append(DataClassAxes("fig1",self.ax1, line, annotations,["O"]))

        #dont show this for if numScenes > 1
        #number 3 (show original outputs for multiple graphs wih multiple y-scale)
        if self.numScenes > 1:
            self.fig2.canvas.manager.set_window_title(f'{self.windowTitle}') 
            self.fig2.suptitle(f"{self.graphTitle} with multiple y-scale", fontsize=12)
            self.ax2.set_ylabel(f"{self.ylabel} {get_folder(self.vistaoutput_path[0])}",color='r')
            self.ax2.tick_params(axis='y', colors='r')
            
            self.ax2.set_xlabel(f"{self.xlabel}")
            self.fig2.tight_layout()   
            self.fig2.set_label('fig2')

            for i in range(self.numScenes):
                if i == 0:
                    #ORIGINAL PLOT
                    line = []
                    line.append(self.ax2.plot(self.xBarData[i][:, 0], self.yBarData[i][:, 0],\
                        f'{colourScheme[np.mod(i,2)][0]}', label=f'Original: {get_folder(self.vistaoutput_path[i])}'))
                    
                    self.fig2.legend()
                    
                    annotations = []
                    self.dcAxes.append(DataClassAxes("fig2",self.ax2, line, annotations,["O"]))
                else:
                    ax2_new = self.ax2.twinx()
                    #ORIGINAL PLOT
                    line = []
                    line.append(ax2_new.plot(self.xBarData[i][:, 0], self.yBarData[i][:, 0],\
                        f'{colourScheme[np.mod(i,2)][0]}', label=f'Original: {get_folder(self.vistaoutput_path[i])}'))
                    #Setting new Y-axis
                    ax2_new.set_ylabel(f"Atomic norm Data rate {get_folder(self.vistaoutput_path[i])}"\
                        , color=f'{colourScheme[np.mod(i,2)][0]}')
                    ax2_new.tick_params(axis='y', colors=f'{colourScheme[np.mod(i,2)][0]}')   
                    
                    offset = (i - 1) * 0.7
                    ax2_new.spines['right'].set_position(('outward', offset * 100))
                    
                    self.fig2.legend()
                    
                    annotations = []
                    self.dcAxes.append(DataClassAxes("fig2",ax2_new, line, annotations,["O"]))
                    
        #number 4.1 (like 2 but with rolling averages)
        self.fig3.canvas.manager.set_window_title(f'{self.windowTitle}') 
        self.fig3.suptitle(f"{self.graphTitle} with only 1 y-scale and average", fontsize=12)
        self.ax3.set_ylabel(f"{self.ylabel} {get_folder(self.vistaoutput_path[0])}",color='r')
        self.ax3.tick_params(axis='y', colors='r') 

        self.ax3.set_xlabel(f"{self.xlabel}")
        self.fig3.tight_layout()
        self.fig3.set_label('fig3')
        
        for i in range(self.numScenes):
            #ORIGINAL PLOT
            line = []
            line.append(self.ax3.plot(self.xBarData[i][:, 0], self.yBarData[i][:, 0],\
                f'{colourScheme[np.mod(i,2)][0]}', label=f'Original: {get_folder(self.vistaoutput_path[i])}', alpha=0.3))
            line.append(self.ax3.plot(self.xBarData[i][:, 0], self.yBarAverageData[i],\
                f'{colourScheme[np.mod(i,2)][1]}', label=f'Rolling Average: {get_folder(self.vistaoutput_path[i])}'))

            self.fig3.legend()
 
            annotations = []
            self.dcAxes.append(DataClassAxes("fig3",self.ax3, line, annotations,["O","A"]))       

        #dont show this for if numScenes > 1
        #number 4.2 (like 3 but with rolling averages)
        if self.numScenes > 1:
            self.fig4.canvas.manager.set_window_title(f'{self.windowTitle}') 
            self.fig4.suptitle(f"{self.graphTitle} with multiple y-scale", fontsize=12)
            self.ax4.set_ylabel(f"{self.ylabel} {get_folder(self.vistaoutput_path[0])}",color='r')
            self.ax4.tick_params(axis='y', colors='r')

            self.ax4.set_xlabel(f"{self.xlabel}")
            self.fig4.tight_layout()              
            self.fig4.set_label('fig4')
            
            for i in range(self.numScenes):
                if i == 0:
                    #ORIGINAL PLOT
                    line = []
                    line.append(self.ax4.plot(self.xBarData[i][:, 0], self.yBarData[i][:, 0],\
                        f'{colourScheme[np.mod(i,2)][0]}', label=f'Original: {get_folder(self.vistaoutput_path[i])}', alpha=0.3))
                    line.append(self.ax4.plot(self.xBarData[i][:, 0], self.yBarAverageData[i],\
                        f'{colourScheme[np.mod(i,2)][1]}', label=f'Rolling Average: {get_folder(self.vistaoutput_path[i])}')) 

                    self.fig4.legend()

                    annotations = []
                    self.dcAxes.append(DataClassAxes("fig4",self.ax4, line, annotations,["O","A"]))  
                else:
                    ax4_new = self.ax4.twinx()
                    #ORIGINAL PLOT
                    line = []
                    line.append(ax4_new.plot(self.xBarData[i][:, 0], self.yBarData[i][:, 0],\
                        f'{colourScheme[np.mod(i,2)][0]}', label=f'Original: {get_folder(self.vistaoutput_path[i])}', alpha=0.3))
                    line.append(ax4_new.plot(self.xBarData[i][:, 0], self.yBarAverageData[i],\
                        f'{colourScheme[np.mod(i,2)][1]}', label=f'Rolling Average: {get_folder(self.vistaoutput_path[i])}'))  
                    #Setting new Y-axis
                    ax4_new.set_ylabel(f"Atomic norm Data rate {get_folder(self.vistaoutput_path[i])}"\
                        , color=f'{colourScheme[np.mod(i,2)][0]}')
                    ax4_new.tick_params(axis='y', colors=f'{colourScheme[np.mod(i,2)][0]}')   
                    
                    offset = (i - 1) * 0.7
                    ax4_new.spines['right'].set_position(('outward', offset * 100))
                    
                    self.fig4.legend()
                        
                    annotations = []
                    self.dcAxes.append(DataClassAxes("fig4",ax4_new, line, annotations,["O","A"]))

    def connect_graph(self):
        print("connect")
        self.fig1.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig1.canvas.mpl_connect('button_press_event', self.on_button_remove)        

        if self.numScenes > 1:
            self.fig2.canvas.mpl_connect('button_press_event', self.on_button_press)
            self.fig2.canvas.mpl_connect('button_press_event', self.on_button_remove)        
    
        self.fig3.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig3.canvas.mpl_connect('button_press_event', self.on_button_remove)
        
        if self.numScenes > 1:
            self.fig4.canvas.mpl_connect('button_press_event', self.on_button_press)
            self.fig4.canvas.mpl_connect('button_press_event', self.on_button_remove)

    def on_button_press(self, event):
        if not event.inaxes:
            return  # off canvas
        #print("hello")
        
        if event.button is MouseButton.LEFT:
            #print("press button: " + str(event.button))
            print((event.canvas.figure).get_label())
            currFig = (event.canvas.figure).get_label()

            all_picked = [dca for dca in self.dcAxes for i in range(len(dca.lines)) if dca.fig == currFig]
            #print(all_picked)
            #all_picked = [dca for dca in self.dcAxes if dca.line.contains(event)[0]]
            if not all_picked:
                return  # nothing was picked

            x_val = event.xdata
            y_val = event.ydata
            
            #currently picks the line belong to the first axis. Can be expanded to include the line
            #along the second axis as well but will need to change downstream code in this function
            for pick in all_picked:
                for idx in range(len(pick.lines)):                
                    ind = (pick.lines[idx][0]).contains(event)[1]['ind']
                    #print("event: " + str((picked.lines[idx][0]).contains(event)))
                    print("first ind: " + str(ind))
                    if len(ind) > 0:
                        #ind_list.append(idx)
                        picked = pick
            #picked = all_picked[0]  # take the first
            print(picked)
            print("x: " + str(event.xdata) + " ,y: " + str(event.ydata))
            '''
            ind = (picked.lines[0][0]).contains(event)[1]['ind']
            #x_index = ind[0]
            print("ind: " +str(ind))
            if not ind:
                ind = (picked.lines[1][0]).contains(event)[1]['ind']
                x_index = ind[0]
                line_index = 1
            else:
                x_index = ind[0]
                line_index = 0

            x_val = (picked.lines[line_index][0]).get_xdata()[x_index]
            y_val = (picked.lines[line_index][0]).get_ydata()[x_index]
            '''
            #x_val = event.xdata
            #y_val = event.ydata
            
            ind_list = []
            for idx in range(len(picked.lines)):
                ind = (picked.lines[idx][0]).contains(event)[1]['ind']
                print("event: " + str((picked.lines[idx][0]).contains(event)))
                print("ind: " + str(ind))
                if len(ind) > 0:
                    ind_list.append(idx)
                    #chosen_line_idx = idx        
            
            x_val, y_val = (picked.ax).transData.inverted().transform([event.x, event.y])
            
            annotation = picked.ax.annotate(
                text='',
                xy=(x_val, y_val),
                xytext=(15, 15),  # distance from x, y
                textcoords='offset points',
                bbox={'boxstyle': 'round', 'fc': 'w'},
                arrowprops={'arrowstyle': '->'}
            )
            #possible bug where there is a point where 2 lines from different axes occur and
            #both read the same value since x and y data is grabbed from 1 event only
            #need to find a way to grab the x and y data from the other axis as well 
            #when this occurs
            text_label = ""
            for idx in ind_list:
                #x_val, y_val = (picked.ax).transData.inverted().transform([event.x, event.y])
                print("x: " + str(x_val) + " ,transformed y: " + str(y_val))
                text_label += f'{picked.lines[idx][0].get_label()}: ({x_val:.2f}, {y_val:.2f})\n'
            annotation.set_text(text_label)

            picked.annotations.append(annotation)
            if currFig == "fig1":
                self.fig1.canvas.draw()
            elif currFig == "fig2":
                self.fig2.canvas.draw()
            elif currFig == "fig3":
                self.fig3.canvas.draw()
            elif currFig == "fig4":
                self.fig4.canvas.draw()
            
    def on_button_remove(self, event):
        if not event.inaxes:
            return  # off canvas

        currFig = (event.canvas.figure).get_label()
        
        if event.button is MouseButton.RIGHT:  # Right-click to remove annotation
            print("remove button: " + str(event.button))
            for dca in self.dcAxes:
                for annotation in dca.annotations:
                    if currFig == "fig1":
                        if annotation.get_window_extent(self.fig1.canvas.get_renderer()).contains(event.x, event.y):
                            annotation.remove()
                            dca.annotations.remove(annotation)
                            self.fig1.canvas.draw_idle()
                            break
                    elif currFig == "fig2":
                        if annotation.get_window_extent(self.fig2.canvas.get_renderer()).contains(event.x, event.y):
                            annotation.remove()
                            dca.annotations.remove(annotation)
                            self.fig2.canvas.draw_idle()
                            break
                    elif currFig == "fig3":
                        if annotation.get_window_extent(self.fig3.canvas.get_renderer()).contains(event.x, event.y):
                            annotation.remove()
                            dca.annotations.remove(annotation)
                            self.fig3.canvas.draw_idle()
                            break
                    elif currFig == "fig4":
                        if annotation.get_window_extent(self.fig4.canvas.get_renderer()).contains(event.x, event.y):
                            annotation.remove()
                            dca.annotations.remove(annotation)
                            self.fig4.canvas.draw_idle()
                            break
    def show(self):
        #plt.legend()
        if self.numScenes > 1:
            plt.show()
        else:
            self.fig1.show()
            self.fig3.show()
            
class InteractiveMethodCompareGraph:
    def __init__(self,idx,xBarData,xBarData2,yBarData,yBarData2,windowTitle,graphTitle,xlabel,ylabel,vistaoutput_path):
        self.idx = idx
        self.fig1, self.ax1 = plt.subplots()
        self.dcAxes = []
        self.annotations = {}
        self.xBarData = xBarData
        self.xBarData2 = xBarData2
        self.yBarData = yBarData
        self.yBarData2 = yBarData2
        self.windowTitle = windowTitle
        self.graphTitle = graphTitle
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.vistaoutput_path = vistaoutput_path
        self.setup_graph()
        self.connect_graph()

    def getGraph(self):
        return self.fig1, self.ax1 

    
    def setup_graph(self):
        # create plots, saving axes/line/annotation for lookup
        @dataclass
        class DataClassAxes:
            fig: str
            ax: plt.Axes
            lines: list[matplotlib.lines.Line2D]
            annotations: list[matplotlib.text.Annotation]
            desc: list[str] #O:Original,A:Average

        self.fig1.canvas.manager.set_window_title(f"{self.windowTitle}: {get_folder(self.vistaoutput_path[self.idx])}") 
        self.fig1.suptitle(f"{self.graphTitle}", fontsize=12)
        
        line = []
        line.append(self.ax1.plot(self.xBarData[self.idx][:, 0], self.yBarData[self.idx],\
        f'r', label=f'Rolling Average of volumetric method: {get_folder(self.vistaoutput_path[self.idx])}'))
        line.append(self.ax1.plot(self.xBarData2[self.idx][:, 0], self.yBarData2[self.idx],\
        f'g', label=f'Rolling Average of simple method: {get_folder(self.vistaoutput_path[self.idx])}'))
          
        self.ax1.set_xlabel(f"{self.xlabel}")
        self.ax1.set_ylabel(f"{self.ylabel}: {get_folder(self.vistaoutput_path[self.idx])}")
        
        self.fig1.legend()
        
        self.fig1.set_label(f"{self.idx}")
        
        annotations = []
        self.dcAxes.append(DataClassAxes(f"{self.idx}",self.ax1, line, annotations,["A"]))


    def connect_graph(self):
        print("connect")
        self.fig1.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.fig1.canvas.mpl_connect('button_press_event', self.on_button_remove)        

    def on_button_press(self, event):
        if not event.inaxes:
            print("nothing")
            return  # off canvas
        print("hello")
        print(event.button)
        
        if event.button is MouseButton.LEFT:
            currFig = (event.canvas.figure).get_label()     

            all_picked = [dca for dca in self.dcAxes for i in range(len(dca.lines)) if dca.fig == currFig]
            #print(all_picked)
            if not all_picked:
                return  # nothing was picked

            x_val = event.xdata
            y_val = event.ydata
            
            #currently picks the line belong to the first axis. Can be expanded to include the line
            #along the second axis as well but will need to change downstream code in this function
            for pick in all_picked:
                for idx in range(len(pick.lines)):                
                    ind = (pick.lines[idx][0]).contains(event)[1]['ind']
                    #print("event: " + str((picked.lines[idx][0]).contains(event)))
                    print("first ind: " + str(ind))
                    if len(ind) > 0:
                        #ind_list.append(idx)
                        picked = pick
            #picked = all_picked[0]  # take the first
            print(picked)
            print("x: " + str(event.xdata) + " ,y: " + str(event.ydata))
            
            ind_list = []
            for idx in range(len(picked.lines)):
                ind = (picked.lines[idx][0]).contains(event)[1]['ind']
                print("event: " + str((picked.lines[idx][0]).contains(event)))
                print("ind: " + str(ind))
                if len(ind) > 0:
                    ind_list.append(idx)
                    #chosen_line_idx = idx        
            
            x_val, y_val = (picked.ax).transData.inverted().transform([event.x, event.y])
            
            annotation = picked.ax.annotate(
                text='',
                xy=(x_val, y_val),
                xytext=(15, 15),  # distance from x, y
                textcoords='offset points',
                bbox={'boxstyle': 'round', 'fc': 'w'},
                arrowprops={'arrowstyle': '->'}
            )
            #possible bug where there is a point where 2 lines from different axes occur and
            #both read the same value since x and y data is grabbed from 1 event only
            #need to find a way to grab the x and y data from the other axis as well 
            #when this occurs
            text_label = ""
            for idx in ind_list:
                #x_val, y_val = (picked.ax).transData.inverted().transform([event.x, event.y])
                print("x: " + str(x_val) + " ,transformed y: " + str(y_val))
                text_label += f'{picked.lines[idx][0].get_label()}: ({x_val:.2f}, {y_val:.2f})\n'
            annotation.set_text(text_label)

            picked.annotations.append(annotation)
            if currFig == f"{self.idx}":
                self.fig1.canvas.draw()
            
    def on_button_remove(self, event):
        if not event.inaxes:
            return  # off canvas

        currFig = (event.canvas.figure).get_label()
        
        if event.button is MouseButton.RIGHT:  # Right-click to remove annotation
            print("remove button: " + str(event.button))
            for dca in self.dcAxes:
                for annotation in dca.annotations:
                    if currFig == f"{self.idx}":
                        if annotation.get_window_extent(self.fig1.canvas.get_renderer()).contains(event.x, event.y):
                            annotation.remove()
                            dca.annotations.remove(annotation)
                            self.fig1.canvas.draw_idle()
                            break

    def show(self):
        self.fig1.show()