import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from properties import data_folder, graphs_folder

cols_to_keep = ["created_date","assignee"]

project_names = [projectfile[:-4] for projectfile in os.listdir(data_folder) if projectfile.endswith(".csv")]
project_names = ["DATALAB"]

for project_name in project_names:
    print("==========================================================================================")
    print(f"Processing: {project_name}")
    print("==========================================================================================")

    # Load the dataset
    df = pd.read_csv(os.path.join(data_folder, project_name + ".csv"), usecols=cols_to_keep)

    # Convert 'created_date' to datetime and organize in weeks
    df['created_date'] = pd.to_datetime(df['created_date'])
    df['week'] = df['created_date'].dt.to_period('W').apply(lambda r: r.start_time)

    # Count the number of issues per assignee per week
    weekly_data = df.groupby(['week', 'assignee']).size().unstack(fill_value=0)
    issues_count = weekly_data.sum(axis=1).to_frame()

    # Dataframe to check wether an assignee was involved or not in a given week
    binary_weekly_data = (weekly_data > 0).astype(int)     

    # Create tick labels for the heatmap axes
    xtick_labels = [f'Week {i}' for i in range(1, len(binary_weekly_data.T.columns) + 1)]
    xtick_labels = [xtick_labels[i] if i % 5 == 0 else '' for i in range(len(xtick_labels))]

    min_val = issues_count.min()[0]
    max_val = issues_count.max()[0]
    mid_val = round((min_val + max_val) / 2)
    
    if not os.path.exists(f'{graphs_folder}/Assignee_Involvement'):
        os.makedirs(f'{graphs_folder}/Assignee_Involvement')

    import matplotlib
    mycmap = "Reds"
    mycmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#f9f4fb","#1F77B4"])
    for i in range(0, 2):
      
        if i == 0:
            fig, ax = plt.subplots(figsize=(12.5, 2.8))
            sns.heatmap(binary_weekly_data.T, vmin=0,vmax=1.25, cmap=mycmap, 
                        cbar= False, xticklabels=xtick_labels, 
                        )
#            ax.set_title(f'Assignee Presence Over Time in {project_name}')
            ax.set_ylabel('Assignee')
            ax.set_xlabel('Week')
            ax.invert_yaxis()
            ax.set_yticklabels(range(1, len(ax.get_yticklabels()) + 1))

            if not os.path.exists(f'{graphs_folder}/{project_name}'):
                os.makedirs(f'{graphs_folder}/{project_name}')
            
            image_name = f'{graphs_folder}/{project_name}/Assignee_Involvement_{project_name}.pdf'
            plt.savefig(image_name, bbox_inches='tight')
            
        else:
            # Create a figure with two subplots
#            fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(14, 4), gridspec_kw={'height_ratios': [7, 1]}, sharex=True)
#            cbar_ax = fig.add_axes([.91, .3, .03, .4])
            gs_kw = dict(width_ratios=[40, 1], height_ratios=[7, 1])
            fig, axd = plt.subplot_mosaic([['upper left', 'right'],
                                           ['lower left', 'right']],
                                          gridspec_kw=gs_kw, figsize=(9.4, 3.1),
                                          layout="constrained")
            ax1, ax2, cbar_ax = axd['upper left'], axd['lower left'], axd['right']
#            cbar_ax.figure.set_size_inches(1, 1)

#            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10), gridspec_kw={'height_ratios': [7, 1]}, sharex=True)
            sns.heatmap(weekly_data.T, cmap=mycmap, xticklabels=False, ax=ax1, vmin=0, vmax=60, cbar = False)
#            ax1.set_title(f'Assignee Involvement Over Time in {project_name}')
            ax1.set_ylabel('Assignee')
            ax1.set_xlabel('')
            ax1.invert_yaxis()
            ax1.set_yticklabels(range(1, len(ax1.get_yticklabels()) + 1))

            # Plot the sum of issues per week in the second subplot
            sns.heatmap(issues_count.T, cmap=mycmap, 
                        xticklabels=xtick_labels, yticklabels=False, 
                        #cbar_kws={'ticks': [min_val, mid_val, max_val]}, 
                        ax=ax2, vmin=0, vmax=60, cbar_ax=cbar_ax
                        )
            ax2.set_xlabel('Week')
            ax2.set_ylabel('Total')
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

            image_name = f'{graphs_folder}/{project_name}/Assignee_Involvement_{project_name}_weighted.pdf'
#            plt.subplots_adjust(wspace=2,hspace=1)
            plt.savefig(image_name, bbox_inches='tight')
        
        #plt.show()
        plt.close()