from util import *

###########################################################################
###########################################################################
# Main function
###########################################################################
###########################################################################
def main():

    # Compute / plot node degrees (sum of all adjacent edge weights)
    if True:
        # Load graph 
        FIn = snap.TFIn(FINAL_UBER_GRAPH_PATH)
        original_graph = snap.TNEANet.Load(FIn)
        # Compute node degree for various attributes
        attributes = ['travel_time', 'travel_speed']
        for attribute in attributes:
            for hour in range(24):
                attribute_hour = attribute + '_' + str(hour)
                new_graph = compute_node_degree(original_graph, attribute_hour)
                # Plot
                draw_map('Data/Geo/Images/%s/%s.png'%(attribute, attribute_hour), \
                            plot_centroids=True, scale_centroids=True, graph=new_graph)

    # Compute average node degree over time
    if False:
        # Load graph 
        FIn = snap.TFIn(FINAL_UBER_GRAPH_PATH)
        original_graph = snap.TNEANet.Load(FIn)
        # Compute node degree for various attributes
        attributes = [('travel_time', 'minutes'), ('travel_speed', 'mph')]
        results = []
        for attribute, label in attributes:
            # Compute
            temp = []
            for hour in range(24):
                new_graph = compute_node_degree(original_graph, attribute+'_'+str(hour), average=True)
                avg_degree = 0
                for node in new_graph.Nodes():
                    avg_degree += new_graph.GetFltAttrDatN(node.GetId(), 'weight')
                avg_degree /= float(new_graph.GetNodes())
                if attribute == 'travel_time': avg_degree /= 60.0
                print('[%d] %.2f' % (hour, avg_degree))
                temp.append(avg_degree)
            # Save
            results.append(temp)
        # Plot
        plt.figure(figsize=(30,20))
        # Time
        fig, ax1 = plt.subplots()
        ax1.plot(range(24), results[0], 'b-')
        ax1.set_ylabel('Avg Travel Time (mins)', color='b')
        ax1.tick_params('y', colors='b')
        # Speed
        ax2 = ax1.twinx()
        ax2.plot(range(24), results[1], 'r-')
        ax2.set_ylabel('Avg Travel Speed (mph)', color='r')
        ax2.tick_params('y', colors='r')
        # Overall
        ax1.set_xlabel('Hour of Day')
        ax1.set_xticks(range(24))
        plt.title('Avg Travel Time vs. Avg Travel Speed for Each Hour of Day')
        plt.savefig('Data/Geo/Images/uber_avg_time_vs_avg_speed.png', dpi=300)

if __name__ == "__main__":
    main()
