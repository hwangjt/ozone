import inspect
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt

from ozone.doc_tests.test_order_plot import Test


Test().test()
plt.savefig('order_plot.pdf')
