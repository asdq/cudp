#include "deviceinfowidget.h"
#include "ui_deviceinfowidget.h"
#include "cudadevice.h"

DeviceInfoWidget::DeviceInfoWidget(int deviceNum, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DeviceInfoWidget)
{
    ui->setupUi(this);
    
    CudaDevice dev(deviceNum);
    ui -> deviceNameValue -> setText(dev.name());
}

DeviceInfoWidget::~DeviceInfoWidget()
{
    delete ui;
}
