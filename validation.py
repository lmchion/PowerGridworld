import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import glob

id_time = '2023-03-26_22-27-12'
filename = glob.glob('data/outputs/ray_results/PPO/PPO_*'+id_time+'*')[0]+'/final_validation.csv'
df = pd.read_csv(filename, index_col=0)

 
timestamp = pd.to_datetime(df['timestamp'], format='%m-%d-%Y %H:%M:%S')
hours = (timestamp.dt.hour + timestamp.dt.minute/60).tolist()
hours = [x if x>=6 else x+24 for x in hours]

with PdfPages('charts/'+ id_time +'.pdf') as pdf:

  plt.figure()

  # Electricity Cost
  supply_grid = (df['es_grid_power_consumed'] + \
            df['ev_grid_power_consumed'] + df['oth_dev_grid_power_consumed'])/12
  plt.plot(hours, (supply_grid * df['grid_price']).cumsum(), label='electricity cost')

  plt.plot(hours, df['es_cost'].cumsum(), label='es_cost')
  plt.plot(hours, df['ev_cost'].cumsum(), label='ev_cost')
  plt.plot(hours, df['oth_dev_cost'].cumsum(), label='oth_dev_cost')
  plt.plot(hours,
          (df['es_cost'] + df['ev_cost'] + df['oth_dev_cost']).cumsum(),
          label = 'total_cost')
  plt.legend(); plt.title('Electricity Cost');
  pdf.savefig(); plt.close()

  # Energy Consumption
  usage_es = (df['es_solar_power_consumed'] + df['es_grid_power_consumed'])/12
  usage_ev = (df['ev_solar_power_consumed'] + \
              df['ev_grid_power_consumed'] + df['ev_es_power_consumed'])/12
  usage_other = (df['oth_dev_solar_power_consumed'] + \
              df['oth_dev_grid_power_consumed'] + df['oth_dev_es_power_consumed'])/12
  plt.plot(hours, usage_es + usage_ev + usage_other, label='usage')
  plt.plot(hours, usage_other, '.', label='usage_other')
  plt.plot(hours, usage_ev, '.', label='usage_ev')
  plt.plot(hours, usage_es, '.', label='usage_es')
  plt.legend(); plt.title('Energy Consumption');
  pdf.savefig(); plt.close()

  # Energy Supply
  supply_solar = (df['es_solar_power_consumed'] + \
            df['ev_solar_power_consumed'] + df['oth_dev_solar_power_consumed'])/12
  supply_es = (df['ev_es_power_consumed'] + df['oth_dev_es_power_consumed'])/12
  supply_grid = (df['es_grid_power_consumed'] + \
            df['ev_grid_power_consumed'] + df['oth_dev_grid_power_consumed'])/12
  plt.plot(hours, supply_solar, '.', label='supply_solar')
  plt.plot(hours, supply_es, '.', label='supply_es')
  plt.plot(hours, supply_grid, '.', label='supply_grid')
  plt.legend(); plt.title('Energy Supply');
  pdf.savefig(); plt.close()

  # Solar
  plt.plot(hours, df['solar_action'], '.', label='solar_action')
  plt.plot(hours, df['solar_available_power'], '.', label='solar_available_power')
  plt.plot(hours, df['solar_actionable_power'], '.', label='solar_actionable_power')
  plt.legend(); plt.title('Solar Power');
  pdf.savefig(); plt.close()
  plt.plot(hours, df['es_solar_power_consumed'], '.', label='es_solar_power_consumed')
  plt.plot(hours, df['ev_solar_power_consumed'], '.', label='ev_solar_power_consumed')
  plt.plot(hours, df['oth_dev_solar_power_consumed'], '.', label='oth_dev_solar_power_consumed')
  plt.legend(); plt.title('Solar Power Usage');
  pdf.savefig(); plt.close()

  # Battery
  plt.plot(hours, df['es_current_storage'], label='es_current_storage')
  plt.plot(hours, df['es_action'], '.', label='es_action')
  plt.legend(); plt.title('Battery Action & Charge');
  pdf.savefig(); plt.close()

  # EV
  plt.plot(hours, df['ev_vehicle_charged'], '.', label='ev_vehicle_charged') # what is the EV charge?
  plt.plot(hours, df['ev_action'], '.', label='ev_action')
  plt.legend(); plt.title('EV Action & Charge');  
  pdf.savefig(); plt.close()
