Modification
============

Add backtest overfitting analyzers.

Example Usage: 
::
   cerebro = bt.Cerebro(optreturn=True, maxcpus=1)
   cerebro.optstrategy(
           BollingerBreakoutStrategy,
           period=range(10, 51, 10),              
           devfactor=[ 1.6,1.8, 2.0,2.3],             
           trend_period=range(25,46,10),            
           size_ratio =[0.05,0.07, 0.1, 0.15, 0.2]         
       )
   #add data 
   #add timereturn analyzer
   cerebro.add_cscv_analyzer(S=16, performance_metric='sharpe')
   results = cerebro.run()
   cscv_results = results['cscv'].results
   print(cscv_results['pbo'])

