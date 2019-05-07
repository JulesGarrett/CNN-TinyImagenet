import sys
import matplotlib.pyplot as plt

def plot_cost_vs_time():
    if len(sys.argv) != 2:
        print("Error! You entered ", len(sys.argv), " arguments but only required 2")
        exit(1)

    file = sys.argv[1]
    inFile = open(file)

    x = []
    y = []

    for line in inFile.readlines():
        line = line.strip()
        if line != "":
            myList = list(map(float, line.split(',')))
            for i in range(0, len(myList)):
                x.append(myList[0])
                y.append(myList[1])

    inFile.close()

    # Plot
    plt.scatter(x, y)
    plt.plot(x, y)
    plt.title('Scatter plot')
    plt.xlabel('Time')
    plt.ylabel('Cost')
    plt.show()


def main():
    plot_cost_vs_time()

if __name__== "__main__":
  main()
