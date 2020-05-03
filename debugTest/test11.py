from __future__ import division, print_function, absolute_import
import pyqtgraph as pg
import pyqtgraph.exporters
import numpy as np

def test_plotscene():
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    w = pg.GraphicsWindow()
    p = w.addPlot()
    p.plot(np.linspace(0.0, 20.0, 20), pen={'color':'b'})
    p.setXRange(0,20)
    p.setYRange(-10,20)
    app = pg.mkQApp()
    app.processEvents()
    ex = pyqtgraph.exporters.ImageExporter(w.scene())
    app.exec()
    #ex.export(fileName="test.png")

if __name__ == "__main__":
    test_plotscene()