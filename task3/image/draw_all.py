import json
import matplotlib.pyplot as plt

plt.figure()
plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['figure.dpi'] = 200
file = ['model_all_diff','model_no_sent','model_normal','model_same_sent']
for i in file:
    with open(i + '.json','r',encoding='utf-8') as file:
        chart_data = json.load(file)
        plt.plot(chart_data['epoch'][:300],chart_data['val_acc'][:300],label=i)
plt.grid(True,axis="y",ls='--')
plt.legend(loc= 'best')
plt.xlabel('epoch',fontsize=20)
# plt.yticks(np.linspace(0,1,11))
plt.savefig('acc.jpg')
plt.close('all')