import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv('final_validation.csv', index_col=0)
#df.head(2).T

timestamp = pd.to_datetime(df['timestamp'], format='%m-%d-%Y %H:%M:%S')
hours = (timestamp.dt.hour + timestamp.dt.minute/60).tolist()
hours = [x if x>=6 else x+24 for x in hours]

with PdfPages('validation.pdf') as pdf:

  plt.figure()

  # solar
  plt.plot(hours, df['pv_power'], '.', label='pv_power')
  plt.plot(hours, df['pv_action'], '.', label='pv_action')
  plt.legend(); pdf.savefig(); plt.close()

  # grid cost
  plt.plot(hours, df['grid_cost'], '.', label='grid_cost')
  plt.legend(); pdf.savefig(); plt.close()

  # battery
  plt.plot(hours, df['es_current_storage'], '.', label='es_current_storage')
  plt.legend(); pdf.savefig(); plt.close()
  plt.plot(hours, df['es_consumed_pv_power'], '.', label='es_consumed_pv_power')
  plt.plot(hours, df['es_consumed_grid_power'], '.', label='es_consumed_grid_power')
  plt.legend(); pdf.savefig(); plt.close()
  plt.plot(hours, df['es_dev_action'], '.', label='es_dev_action')
  plt.legend(); pdf.savefig(); plt.close()
  plt.plot(hours, df['es_cost'], '.', label='es_cost')
  plt.plot(hours, df['es_reward'], '.', label='es_reward')
  plt.plot(hours, df['es_current_psudo_cost'], '.', label='es_current_psudo_cost')
  plt.legend(); pdf.savefig(); plt.close()

  # EV
  plt.plot(hours, df['ev_power_unserved'], '.', label='ev_power_unserved')
  plt.legend(); pdf.savefig(); plt.close()
  plt.plot(hours, df['ev_consumed_es_power'], '.', label='ev_consumed_es_power')
  plt.plot(hours, df['ev_consumed_pv_power'], '.', label='ev_consumed_pv_power')
  plt.plot(hours, df['ev_consumed_grid_power'], '.', label='ev_consumed_grid_power')
  plt.legend(); pdf.savefig(); plt.close()
  plt.plot(hours, df['ev_dev_action'], '.', label='ev_dev_action')
  plt.legend(); pdf.savefig(); plt.close()
  plt.plot(hours, df['ev_cost'], '.', label='ev_cost')
  plt.plot(hours, df['ev_reward'], '.', label='ev_reward')
  plt.legend(); pdf.savefig(); plt.close()

  # Other
  plt.plot(hours, df['oth_dev_consumed_es_power'], '.', label='oth_dev_consumed_es_power')
  plt.plot(hours, df['oth_dev_consumed_pv_power'], '.', label='oth_dev_consumed_pv_power')
  plt.plot(hours, df['oth_dev_consumed_grid_power'], '.', label='oth_dev_consumed_grid_power')
  plt.legend(); pdf.savefig(); plt.close()
  plt.plot(hours, df['oth_dev_action'], '.', label='oth_dev_action')
  plt.legend(); pdf.savefig(); plt.close()
  plt.plot(hours, df['oth_dev_cost'], '.', label='oth_dev_cost')
  plt.plot(hours, df['oth_dev_reward'], '.', label='oth_dev_reward')
  plt.legend(); pdf.savefig(); plt.close()
