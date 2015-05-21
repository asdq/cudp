#ifndef DEVICEINFOWIDGET_H
#define DEVICEINFOWIDGET_H

#include <QWidget>
#include "cudadevice.h"

namespace Ui {
    class DeviceInfoWidget;
}

class DeviceInfoWidget : public QWidget
{
    Q_OBJECT
    
public:
    
    explicit
    DeviceInfoWidget(QWidget *parent = 0);
    
    ~DeviceInfoWidget();
    
    void setData(const CudaDevice &dev);
    
private:
    Ui::DeviceInfoWidget *ui;
};

#endif // DEVICEINFOWIDGET_H
