#include "deviceinfowidget.h"
#include "ui_deviceinfowidget.h"
#include "cudadevice.h"
#include <QStringBuilder>

DeviceInfoWidget::DeviceInfoWidget(int deviceNum, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::DeviceInfoWidget)
{
    ui->setupUi(this);
    
    CudaDevice dev(deviceNum);
    
    ui -> deviceNameValue -> setText(dev.name());
    ui -> deviceAsyncValue -> setText(QString::number(dev.asyncEngineCount()));
    ui -> deviceNumberValue -> setText(QString::number(deviceNum));
    
    ui -> deviceOverlappingValue
    -> setText(dev.deviceOverlap() ? tr("Yes") : tr("No"));
    
    ui -> deviceMapValue
    -> setText(dev.deviceMapHost() ? tr("Yes") : tr("No"));
    
    ui -> deviceVersionValue
    -> setText(QString::number(dev.majorVersion()) % "."
               % QString::number(dev.minorVersion()));
    
    ui -> memConstantValue
    -> setText(QString("%L1 bytes").arg(dev.totalConstMem()));
    
    ui -> memGlobalValue
    -> setText(QString("%L1 bytes").arg(dev.totalGlobalMem()));
    
    ui -> memSharedValue
    -> setText(QString("%L1 bytes").arg(dev.sharedMemPerBlock()));
    
    ui -> threadBlockValue
    -> setText(QString("%L1").arg(dev.maxThreadsPerBlock()));
    
    ui -> threadGridValue
    -> setText("(" % QString("%L1").arg(dev.maxGridSizeX()) % ", "
               % QString("%L1").arg(dev.maxGridSizeY()) % ", "
               % QString("%L1").arg(dev.maxThreadsDimZ()) % ")");
    
    ui -> threadMultiprocValue
    -> setText(QString("%L1").arg(dev.maxThreadsPerMultiProcessor()));
    
    ui -> threadSizeValue
    -> setText("(" % QString("%L1").arg(dev.maxThreadsDimX()) % ", "
               % QString("%L1").arg(dev.maxThreadsDimY()) % ", "
               % QString("%L1").arg(dev.maxThreadsDimZ()) % ")");
    
    ui -> threadWarpValue -> setText(QString::number(dev.warpSize()));
}

DeviceInfoWidget::~DeviceInfoWidget()
{
    delete ui;
}
