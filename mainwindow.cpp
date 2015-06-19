#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "deviceinfowidget.h"
#include <QMessageBox>

extern "C" {
#include <cuda_runtime_api.h>
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    m_messageBox = new QMessageBox(this);
    ui->setupUi(this);
    initHelp();
    addDevices();
}

MainWindow::~MainWindow()
{
    delete ui;
}

/*
 * @brief Initialize Help dialogs.
 */
void MainWindow::initHelp()
{
    QString txt = tr("<p align=\"center\">"
                     "<b>Display CUDA devices version 0.3</b></p>"
                     "<p align=\"justify\">Lookup for CUDA devices. Display "
                     "in a tab the properties of each device found. See "
                     "<a href=\"https://github.com/asdq/qtcudaprop\">"
                     "github.com/asdq/qtcudaprop</a>.</p>"
                     "<p align=\"right\"><b>Author: Fabio Vaccari</b></p>"
                     "<p align=\"right\"><b>License: MIT</b><p>"
                     );
    
    m_messageBox -> setTextFormat(Qt::RichText);
    m_messageBox -> setWindowTitle(tr("About QtCUDAProperties"));
    m_messageBox -> setText(txt);
    
    connect(ui -> actionAbout, SIGNAL(triggered()), m_messageBox, SLOT(exec()));
    connect(ui -> actionAboutQt, SIGNAL(triggered()), qApp, SLOT(aboutQt()));
}

/*
 * @brief Create a tab for each device.
 */
void MainWindow::addDevices()
{
    cudaError_t err;
    int numDevices;
    
    err = cudaGetDeviceCount(&numDevices);
    
    if (err != cudaSuccess) {
        QMessageBox *msgBox = new QMessageBox(this);
        QString txt;
        
        if (err == cudaErrorNoDevice) {
            txt = tr("CUDA Error: No device found.");
        } else if (err == cudaErrorInsufficientDriver) {
            txt = tr("CUDA Error: Insufficient driver.");
        } else { 
            txt = tr("CUDA Error: Unhandled error %L1").arg(err);
        }
        msgBox -> setText(txt);
        msgBox -> show();
        return;
    }
    
    for (int i = 0; i < numDevices; ++i) {
        cudaDeviceProp dev;
        
        err = cudaGetDeviceProperties(&dev, i);
        if (err == cudaSuccess) {
            DeviceInfoWidget *dw = new DeviceInfoWidget;
            QString label = QString("Device &%L1").arg(i);
            
            dw -> setData(&dev);
            ui -> tabWidget -> addTab(dw, label);
        } else {
            QMessageBox *msgBox = new QMessageBox(this);
            
            msgBox -> setModal(false);
            msgBox -> setText(tr("Invalid device %L1.").arg(i));
            msgBox -> show();
        }
    }
}
