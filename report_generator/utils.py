# ------------------------------------------------------------------------------
#
#
#                                 P̶̖̥͈͈͇̼͇̈́̈́̀͋͒̽͊͘͠h̴͙͈̦͗́̓a̴̧͗̾̀̅̇ḡ̶͓̭̝͓͎̰͓̦̭̎́͊̐̂͒͠ơ̵̘͌̿̽̑̈̾Ś̴̠́̓̋̂̃͒͆̚t̴̲͓̬͎̾͑͆͊́̕a̸͍̫͎̗̞͆̇̀̌̑͂t̸̳̼̘͕̯̠̱̠͂̔̒̐̒̕͝͝
#
#
#                                PhagoStat
#                Advanced Phagocytic Activity Analysis Tool
# ------------------------------------------------------------------------------
# Copyright (C) 2023 Mehdi OUNISSI <mehdi.ounissi@icm-institute.org>
#               Sorbonne University, Paris Brain Institute - ICM, CNRS, Inria,
#               Inserm, AP-HP, Paris, 75013, France.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ------------------------------------------------------------------------------
# Note on Imported Packages:
# The packages used in this work are imported as is and not modified. If you
# intend to use, modify, or distribute any of these packages, please refer to
# the requirements.txt file and the respective package licenses.
# ------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use('Agg')


def phagocytosis_evolution_to_csv_pdf(phagocytosis_evolution_csv_path,
                                      cell_tracking_report_csv_path,
                                      output_path,
                                      keep_until=200,
                                      time_per_frame=2):
    # Function to remove negative values
    def remove_zeros(tab):
        for i in range(len(tab)):
            if tab[i]< 0: tab[i]=0
        return tab
    
    # Reading the csv file in data frame format
    df = pd.read_csv(phagocytosis_evolution_csv_path)
    
    # Retreaving the data frame keys
    aggr_keys = df.keys()

    # Reading the time stamps
    time = list(df[aggr_keys[0]])

    # Check if the document have enough data points
    if time[-1]>= keep_until:

        # Reading the aggregate count into list format
        aggr_count = list(df[aggr_keys[1]])

        # Reading the aggregate area into list format
        aggr_area = list(df[aggr_keys[2]])    
        
        # Initiate a dummy counter
        list_idx = 0
        
        # Initiate list to store interpolated data
        new_time, new_aggr_count, new_aggr_area = [],[],[]

        tmp_time = np.arange(0, keep_until+time_per_frame, time_per_frame)
        
        # Gp over all the time points
        for i in range(len(tmp_time)):
            # Check if the presence of data in the time point
            if tmp_time[i] == time[list_idx]:
                new_time.append(time[list_idx])
                new_aggr_count.append(aggr_count[list_idx])
                new_aggr_area.append(aggr_area[list_idx])

                # Incrementing the list index counter
                list_idx += 1
            
            # Filling time points with np.nan (data to be interpolated)
            else:
                new_time.append(tmp_time[i])
                new_aggr_count.append(np.nan)
                new_aggr_area.append(np.nan)
        
        new_aggr_count = remove_zeros(new_aggr_count)
        new_aggr_area = remove_zeros(new_aggr_area)
        
        # Dictionary lists of data with np.nan at the gaps
        dict = {aggr_keys[0]:new_time,
                aggr_keys[1]:new_aggr_count,
                aggr_keys[2]:new_aggr_area}
        
        # Creating a dataframe from dictionary
        new_df = pd.DataFrame(dict)

        # Using forward linear interpolation to fill np.nan (gaps) 
        intr_new_df = new_df.interpolate(method ='linear', limit_direction ='forward')
        
        # Getting the corrected data frame data keys
        aggr_keys = intr_new_df.keys()

        # Retreving the time list
        time = list(intr_new_df[aggr_keys[0]])

        # Retreving the aggregate count list
        aggr_count = list(intr_new_df[aggr_keys[1]])

        # Retreving the aggregate area list
        aggr_area = list(intr_new_df[aggr_keys[2]])

        # Reading the csv file in data frame format
        df = pd.read_csv(cell_tracking_report_csv_path)

        # Retreaving the data frame keys
        cell_keys = df.keys()

        # Defining the stop list index
        stop_index = int(keep_until/time_per_frame)+1

        # Reading the time stamps
        time = list(df[cell_keys[0]])[0:stop_index]

        # Reading the cell's mean speed into list format
        cell_mean_speed = list(df[cell_keys[1]])[0:stop_index]

        # Reading the cell's mean area via tracking into list format
        cell_mean_area_tk = list(df[cell_keys[2]])[0:stop_index]

        # Reading the cell's mean area via total area divided by cell count into list format
        cell_mean_area = list(df[cell_keys[3]])[0:stop_index]

        # Reading the cell's total area into list format
        cell_total_area = list(df[cell_keys[4]])[0:stop_index]

        # Reading the cell's count into list format
        cell_count = list(df[cell_keys[5]])[0:stop_index]

        # Reading the cell's total movement into list format
        cell_total_move = list(df[cell_keys[6]])[0:stop_index]

        # Computting the ratio (total area eaten/total cell area)
        aggr_ratio_area = list(np.array(aggr_area) / np.array(cell_total_area))

        # Computting the ratio (total area eaten/total cell area)
        aggr_ratio_count = list(np.array(aggr_area) / np.array(cell_count))

        ############# Preprating the clean data in csv file ########################

        # Dictionary lists of all phagocytosis features data
        dict = {cell_keys[0]                                           :time,
                'Aggregates ratio (area eaten/cell count) (microns^2)' :aggr_ratio_count,
                'Aggregates ratio (area eaten/cell area)'              :aggr_ratio_area,
                aggr_keys[1]                                           :aggr_count,
                aggr_keys[2]                                           :aggr_area,
                cell_keys[1]                                           :cell_mean_speed,
                cell_keys[2]                                           :cell_mean_area_tk,
                cell_keys[3]                                           :cell_mean_area,
                cell_keys[4]                                           :cell_total_area,
                cell_keys[5]                                           :cell_count,
                cell_keys[6]                                           :cell_total_move}

        
        # Creating a dataframe from dictionary
        clean_df = pd.DataFrame(dict)

        # Getting the file direcotires
        scene_dir           = os.path.dirname(phagocytosis_evolution_csv_path)
        acquisition_id_dir  = os.path.dirname(scene_dir)
        genotype_dir        = os.path.dirname(acquisition_id_dir)

        # Getting key name to store reports
        scene_name          = os.path.basename(scene_dir)
        acquisition_id_name = os.path.basename(acquisition_id_dir)
        genotype_name       = os.path.basename(genotype_dir)

        # Defining the path where to store figure
        figures_output_dir = os.path.join(output_path, genotype_name, acquisition_id_name, scene_name+'_figures')
        
        # Creating a directory to store figures
        os.makedirs(figures_output_dir, exist_ok = True)

        # Saving the data frame
        clean_df.to_csv(os.path.join(output_path, genotype_name, acquisition_id_name, scene_name+'.csv'), index = False)

        # Retreaving the data frame keys
        clean_df_keys = clean_df.keys()

        # # Plotting the data
        # for i in range(1,len(clean_df_keys)):
        #     plt.figure(figsize=(8,6))
        #     plt.plot(list(clean_df[clean_df_keys[0]]), list(clean_df[clean_df_keys[i]]))
        #     plt.xlabel("Time [min]")
        #     plt.xlim((0, np.max(list(clean_df[clean_df_keys[0]]))))
        #     plt.ylim((0, np.max(list(clean_df[clean_df_keys[i]]))))
        #     plt.ylabel(clean_df_keys[i])
        #     plt.tight_layout()
        #     plt.savefig(os.path.join(output_path, scene_name+'_figures', str(i)+'.pdf'), dpi=200)
        #     plt.close()
        
        for i in range(1,len(clean_df_keys)):
            # Get the Peaks and Troughs
            data = clean_df[clean_df_keys[i]].values
            doublediff = np.diff(np.sign(np.diff(data)))
            peak_locations = np.where(doublediff == -2)[0] + 1

            doublediff2 = np.diff(np.sign(np.diff(-1*data)))
            trough_locations = np.where(doublediff2 == -2)[0] + 1

            # Draw Plot
            plt.figure(figsize=(16,10), dpi= 80)
            plt.plot(clean_df_keys[0], clean_df_keys[i], data=clean_df, color='tab:blue', label=clean_df_keys[i])
            plt.scatter(clean_df[clean_df_keys[0]][peak_locations], clean_df[clean_df_keys[i]][peak_locations], marker=matplotlib.markers.CARETUPBASE, color='tab:green', s=100, label='Peaks')
            plt.scatter(clean_df[clean_df_keys[0]][trough_locations], clean_df[clean_df_keys[i]][trough_locations], marker=matplotlib.markers.CARETDOWNBASE, color='tab:red', s=100, label='Troughs')

            # Annotate
            for t, p in zip(trough_locations[1::4], peak_locations[::3]):
                plt.text(clean_df[clean_df_keys[0]][p], clean_df[clean_df_keys[i]][p]+0.05*clean_df[clean_df_keys[i]][p], round(clean_df[clean_df_keys[i]][p],4), horizontalalignment='center', color='darkgreen')
                plt.text(clean_df[clean_df_keys[0]][t], clean_df[clean_df_keys[i]][t]-0.05*clean_df[clean_df_keys[i]][t], round(clean_df[clean_df_keys[i]][t],4), horizontalalignment='center', color='darkred')

            # Decoration
            plt.xlim((0, np.max(list(clean_df[clean_df_keys[0]]))))
            plt.ylim((0, np.max(list(clean_df[clean_df_keys[i]]))+0.1*np.max(list(clean_df[clean_df_keys[i]]))))

            plt.title(clean_df_keys[i], fontsize=22)
            plt.yticks(fontsize=12, alpha=.7)

            # Lighten borders
            plt.gca().spines["top"].set_alpha(.0)
            plt.gca().spines["bottom"].set_alpha(.3)
            plt.gca().spines["right"].set_alpha(.0)
            plt.gca().spines["left"].set_alpha(.3)

            plt.legend(loc='upper left')
            plt.grid(axis='y', alpha=.3)
            plt.savefig(os.path.join(figures_output_dir, str(i)+'.pdf'), dpi=200)
            plt.close()
        

def plot_all_scenes(scenes_csv_paths, output_path):
    # Reading the first csv file to retreive the colum names
    first_df = pd.read_csv(scenes_csv_paths[0])
    col_names = first_df.keys()


    # Plotting the data
    for i in range(1,len(col_names)):
        max_data = 0
        plt.figure(figsize=(8,6))
        for path in scenes_csv_paths:
            # Reading the csv file
            tmp_df = pd.read_csv(path)

            # Looking for max value (for nice plots)
            tmp_max = np.max(list(tmp_df[col_names[i]]))
            
            if tmp_max > max_data: max_data = tmp_max

            # Getting the file name without extention
            scene_label = os.path.splitext(os.path.basename(path))[0]

            # Plotting the current csv file's data 
            plt.plot(list(tmp_df[col_names[0]]), list(tmp_df[col_names[i]]), label=scene_label)
        
        plt.xlabel("Time [min]")
        plt.xlim((0, np.max(list(tmp_df[col_names[0]]))))
        plt.ylim((0, max_data+0.1*max_data))
        plt.ylabel(col_names[i])
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.yticks(fontsize=12, alpha=.7)

        

        
        plt.title(col_names[i])
        # Lighten borders
        plt.gca().spines["top"].set_alpha(.0)
        plt.gca().spines["bottom"].set_alpha(.3)
        plt.gca().spines["right"].set_alpha(.0)
        plt.gca().spines["left"].set_alpha(.3)
        plt.grid(axis='y', alpha=.3)

        plt.tight_layout()

        
        plt.savefig(os.path.join(output_path, str(i)+'.pdf'), dpi=200)
        plt.close()


def plot_mean_scenes(mean_df, std_df, plot_label,output_path):
    # Retreiving the colum names
    col_names = mean_df.keys()

    # Plotting the data
    for i in range(1,len(col_names)):
        

        # Plotting the current csv file's data 
        plt.figure(figsize=(8,6))
        plt.plot(list(mean_df[col_names[0]]), list(mean_df[col_names[i]]), color='tab:blue', label=plot_label)
        plt.fill_between(list(mean_df[col_names[0]]),
                         list(mean_df[col_names[i]]-std_df[col_names[i]]),
                         list(mean_df[col_names[i]]+std_df[col_names[i]]),
                         alpha=0.1, linewidth=0, color='tab:blue')
        
        plt.xlabel("Time [min]")
        plt.xlim((0, np.max(list(mean_df[col_names[0]]))))
        plt.ylim((0, np.max(list(mean_df[col_names[i]]))+0.2*np.max(list(mean_df[col_names[i]]))))
        
        
        plt.yticks(fontsize=12, alpha=.7)
        plt.ylabel(col_names[i])
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        

        
        plt.title(col_names[i])
        # Lighten borders
        plt.gca().spines["top"].set_alpha(.0)
        plt.gca().spines["bottom"].set_alpha(.3)
        plt.gca().spines["right"].set_alpha(.0)
        plt.gca().spines["left"].set_alpha(.3)
        plt.grid(axis='y', alpha=.3)

        plt.tight_layout()
        
        plt.savefig(os.path.join(output_path, str(i)+'.pdf'), dpi=200)
        plt.close()

def csv_list_to_mean_std(scenes_csv_paths, file_name, data_output_dir):
    # Reading the first csv file to retreive the colum names
    first_df = pd.read_csv(scenes_csv_paths[0])
    col_names = first_df.keys()

    # Initiate a list to store all the data
    csv_list = []
    for csv_path in scenes_csv_paths:
        csv_list.append(np.array(pd.read_csv(csv_path)))

    # Converting the list to numpy array
    csv_list = np.array(csv_list)

    # Computting the mean of the data frames
    mean_np = np.mean(csv_list, axis=0)

    # Computting the std of the data frames
    std_np = np.std(csv_list, axis=0)

    # Correvting the time colum
    std_np[:,0] = mean_np[:,0]

    # Converting the data from numpy array to pandas data frame
    mean_df = pd.DataFrame(data=mean_np[0:,0:],
                           columns=[col_names[i] for i in range(mean_np.shape[1])])
    
    std_df  = pd.DataFrame(data=std_np[0:,0:],
                           columns=[col_names[i] for i in range(std_np.shape[1])])


    # Saving the data frame
    mean_df.to_csv(os.path.join(data_output_dir, file_name+'_mean.csv'), index = False)
    std_df.to_csv( os.path.join(data_output_dir, file_name+'_std.csv'), index = False)

    return mean_df, std_df

def add_p_value_annotation(fig, array_columns, subplot=None, _format=dict(interline=0.07, text_height=1.07, color='black')):
    ''' Adds notations giving the p-value between two box plot data (t-test two-sided comparison)
    
    Parameters:
    ----------
    fig: figure
        plotly boxplot figure
    array_columns: np.array
        array of which columns to compare 
        e.g.: [[0,1], [1,2]] compares column 0 with 1 and 1 with 2
    subplot: None or int
        specifies if the figures has subplots and what subplot to add the notation to
    _format: dict
        format characteristics for the lines

    Returns:
    -------
    fig: figure
        figure with the added notation
    '''
    # Specify in what y_range to plot for each pair of columns
    y_range = np.zeros([len(array_columns), 2])
    for i in range(len(array_columns)):
        y_range[i] = [1.01+i*_format['interline'], 1.02+i*_format['interline']]

    # Get values from figure
    fig_dict = fig.to_dict()

    # Get indices if working with subplots
    if subplot:
        if subplot == 1:
            subplot_str = ''
        else:
            subplot_str =str(subplot)
        indices = [] #Change the box index to the indices of the data for that subplot
        for index, data in enumerate(fig_dict['data']):
            #print(index, data['xaxis'], 'x' + subplot_str)
            if data['xaxis'] == 'x' + subplot_str:
                indices = np.append(indices, index)
        indices = [int(i) for i in indices]
        print((indices))
    else:
        subplot_str = ''

    # Print the p-values
    for index, column_pair in enumerate(array_columns):
        if subplot:
            data_pair = [indices[column_pair[0]], indices[column_pair[1]]]
        else:
            data_pair = column_pair

        # Mare sure it is selecting the data and subplot you want
        #print('0:', fig_dict['data'][data_pair[0]]['name'], fig_dict['data'][data_pair[0]]['xaxis'])
        #print('1:', fig_dict['data'][data_pair[1]]['name'], fig_dict['data'][data_pair[1]]['xaxis'])

        pvalue = mannwhitneyu(
            fig_dict['data'][data_pair[0]]['y'],
            fig_dict['data'][data_pair[1]]['y']
            )[1]


        if pvalue >= 0.05:
            symbol = 'ns'
        elif pvalue >= 0.01: 
            symbol = '*'
        elif pvalue >= 0.001:
            symbol = '**'
        else:
            symbol = '***'
        # Vertical line
        fig.add_shape(type="line",
            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
            x0=column_pair[0], y0=y_range[index][0], 
            x1=column_pair[0], y1=y_range[index][1],
            line=dict(color=_format['color'], width=2,)
        )
        # Horizontal line
        fig.add_shape(type="line",
            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
            x0=column_pair[0], y0=y_range[index][1], 
            x1=column_pair[1], y1=y_range[index][1],
            line=dict(color=_format['color'], width=2,)
        )
        # Vertical line
        fig.add_shape(type="line",
            xref="x"+subplot_str, yref="y"+subplot_str+" domain",
            x0=column_pair[1], y0=y_range[index][0], 
            x1=column_pair[1], y1=y_range[index][1],
            line=dict(color=_format['color'], width=2,)
        )
        ## add text at the correct x, y coordinates
        ## for bars, there is a direct mapping from the bar number to 0, 1, 2...
        fig.add_annotation(dict(font=dict(color=_format['color'],size=14),
            x=(column_pair[0] + column_pair[1])/2,
            y=y_range[index][1]*_format['text_height'],
            showarrow=False,
            text=symbol,
            textangle=0,
            xref="x"+subplot_str,
            yref="y"+subplot_str+" domain"
        ))
    return fig