#include "deviceinfowidget.h"
#include "ui_deviceinfowidget.h"

DeviceInfoWidget::DeviceInfoWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DeviceInfoWidget)
{
    ui->setupUi(this);
}

void DeviceInfoWidget::setData(const CudaDevice &dev)
{
    ui -> deviceNameValue -> setText(dev.name());
    ui -> deviceAsyncValue -> setText(QString(tr("%L1"))
                                      .arg(dev.asyncEngineCount()));
    
    ui -> deviceOverlappingValue
       -> setText(dev.deviceOverlap() ? tr("Yes") : tr("No"));
    
    ui -> deviceMapValue
       -> setText(dev.deviceMapHost() ? tr("Yes") : tr("No"));
    
    ui -> deviceVersionValue
       -> setText(QString(tr("%L1.%L2")).arg(dev.majorVersion())
               .arg(dev.minorVersion()));
    
    ui -> memConstantValue
       -> setText(QString(tr("%L1 bytes")).arg(dev.totalConstMem()));
    
    ui -> memGlobalValue
       -> setText(QString(tr("%L1 bytes")).arg(dev.totalGlobalMem()));
    
    ui -> memSharedValue
       -> setText(QString(tr("%L1 bytes")).arg(dev.sharedMemPerBlock()));
    
    ui -> threadBlockValue
       -> setText(QString(tr("%L1")).arg(dev.maxThreadsPerBlock()));
    
    ui -> threadGridValue
       -> setText(QString(tr("(%L1, %L2, %L3)")).arg(dev.maxGridSizeX())
                  .arg(dev.maxGridSizeY()).arg(dev.maxThreadsDimZ()));
    
    ui -> threadMultiprocValue
       -> setText(QString(tr("%L1")).arg(dev.maxThreadsPerMultiProcessor()));
    
    ui -> threadSizeValue
       -> setText(QString(tr("(%L1, %L2, %L3)")).arg(dev.maxThreadsDimX())
                  .arg(dev.maxThreadsDimY()).arg(dev.maxThreadsDimZ()));
    
    ui -> threadWarpValue -> setText(QString(tr("%L1")).arg(dev.warpSize()));
}

DeviceInfoWidget::~DeviceInfoWidget()
{
    delete ui;
}
