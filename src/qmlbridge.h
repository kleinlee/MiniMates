#ifndef QMLBRIDGE_H
#define QMLBRIDGE_H
#include <QQmlEngine>
#include <QObject>
#include <QProcess>
#include "humanassets.h"
class QmlBridge : public QObject
{
    Q_OBJECT
    QML_NAMED_ELEMENT(QmlBridge)
    Q_PROPERTY(bool llmLoaded READ llmLoaded NOTIFY llmLoadedChanged)

public:
    QmlBridge();

    Q_INVOKABLE int loadLLMModel();
    Q_INVOKABLE void setVoiceName(int index);

    bool llmLoaded() const { return m_llm_loaded; }

public:
    HumanAssets* human_assets_;
    bool m_llm_loaded = false;

signals:
    void processStartedSuccessfully();
    void processStartFailed();
    void llmLoadedChanged();

private:
    QProcess m_llm_process;
    void handleProcessOutput();
};

#endif // QMLBRIDGE_H
