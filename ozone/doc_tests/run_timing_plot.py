import inspect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from ozone.doc_tests.test_timing_plot import Test


Test().test()
plt.savefig('timing_plot.pdf')
