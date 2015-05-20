#ifndef DEVICEINFOWIDGET_H
#define DEVICEINFOWIDGET_H

#include <QWidget>

namespace Ui {
    class DeviceInfoWidget;
}

class DeviceInfoWidget : public QWidget
{
    Q_OBJECT
    
public:
    
    explicit
    DeviceInfoWidget(int deviceNum, QWidget *parent = 0);
    
    ~DeviceInfoWidget();
    
private:
    Ui::DeviceInfoWidget *ui;
};

#endif // DEVICEINFOWIDGET_H
