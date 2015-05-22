#include "deviceinfowidget.h"
#include "ui_deviceinfowidget.h"

extern "C" {
#include <cuda_runtime_api.h>
}

DeviceInfoWidget::DeviceInfoWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DeviceInfoWidget)
{
    ui->setupUi(this);
}

void DeviceInfoWidget::setData(cudaDeviceProp *dev)
{
    ui -> deviceNameValue -> setText(dev -> name);
    ui -> deviceAsyncValue -> setText(QString(tr("%L1"))
                                      .arg(dev -> asyncEngineCount));
    
    ui -> deviceOverlappingValue
       -> setText((dev -> deviceOverlap > 0) ? tr("Yes") : tr("No"));
    
    ui -> deviceMapValue
       -> setText((dev -> canMapHostMemory > 0) ? tr("Yes") : tr("No"));
    
    ui -> deviceVersionValue
       -> setText(QString(tr("%L1.%L2")).arg(dev -> major).arg(dev -> minor));
    
    ui -> memConstantValue
       -> setText(QString(tr("%L1 bytes")).arg(dev -> totalConstMem));
    
    ui -> memGlobalValue
       -> setText(QString(tr("%L1 bytes")).arg(dev -> totalGlobalMem));
    
    ui -> memSharedValue
       -> setText(QString(tr("%L1 bytes")).arg(dev -> sharedMemPerBlock));
    
    ui -> threadBlockValue
       -> setText(QString(tr("%L1")).arg(dev -> maxThreadsPerBlock));
    
    ui -> threadGridValue
       -> setText(QString(tr("(%L1, %L2, %L3)")).arg(dev -> maxGridSize[0])
                  .arg(dev -> maxGridSize[1]).arg(dev -> maxGridSize[2]));
    
    ui -> threadMultiprocValue
       -> setText(QString(tr("%L1")).arg(dev -> maxThreadsPerMultiProcessor));
    
    ui -> threadSizeValue
       -> setText(QString(tr("(%L1, %L2, %L3)")).arg(dev -> maxThreadsDim[0])
                  .arg(dev -> maxThreadsDim[1]).arg(dev -> maxThreadsDim[2]));
    
    ui -> threadWarpValue -> setText(QString(tr("%L1")).arg(dev -> warpSize));
}

DeviceInfoWidget::~DeviceInfoWidget()
{
    delete ui;
}
