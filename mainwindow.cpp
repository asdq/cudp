#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "deviceinfowidget.h"
#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    
    const int numDevices = CudaDevice::deviceCount();
    
    for (int i = 0; i < numDevices; ++i) {
        try {
            CudaDevice dev(i);
            auto dw = new DeviceInfoWidget;
            auto label = QString("Device &%1").arg(i);
            
            dw -> setData(dev);
            ui -> tabWidget -> addTab(dw, label);
        } catch (CudaDevice::Exception e) {
            auto msgBox = new QMessageBox(this);
            
            msgBox -> setModal(false);
            msgBox -> setText(e.what());
            msgBox -> show();
        }
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}
