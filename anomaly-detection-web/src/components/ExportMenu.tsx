import { Download, Share2, FileText, Image, Copy } from 'lucide-react';
import { useState } from 'react';
import { ChartExporter } from '../utils/chartExporter';

interface ExportMenuProps {
    chartId: string;
    chartTitle: string;
    data?: any[];
    headers?: string[];
}

export function ExportMenu({ chartId, chartTitle, data, headers }: ExportMenuProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [isExporting, setIsExporting] = useState(false);

    const handleExport = async (type: 'png' | 'pdf' | 'csv' | 'share' | 'copy') => {
        setIsExporting(true);
        try {
            switch (type) {
                case 'png':
                    await ChartExporter.exportAsPNG(chartId, `${chartTitle}.png`);
                    break;
                case 'pdf':
                    await ChartExporter.exportAsPDF(chartId, `${chartTitle}.pdf`);
                    break;
                case 'csv':
                    if (data && headers) {
                        ChartExporter.exportAsCSV(data, headers, `${chartTitle}.csv`);
                    }
                    break;
                case 'share':
                    await ChartExporter.shareChart(chartId, chartTitle);
                    break;
                case 'copy':
                    await ChartExporter.copyToClipboard(chartId);
                    break;
            }
            setIsOpen(false);
        } catch (error) {
            console.error('Export error:', error);
            alert('Export failed. Please try again.');
        } finally {
            setIsExporting(false);
        }
    };

    return (
        <div className="relative">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="size-8 rounded-lg bg-surface border border-white/5 flex items-center justify-center text-secondary hover:text-primary hover:bg-surfaceHover transition-colors"
                disabled={isExporting}
            >
                <Download size={16} />
            </button>

            {isOpen && (
                <>
                    {/* Backdrop */}
                    <div
                        className="fixed inset-0 z-40"
                        onClick={() => setIsOpen(false)}
                    />

                    {/* Menu */}
                    <div className="absolute right-0 mt-2 w-48 bg-surface border border-white/10 rounded-xl shadow-xl z-50 overflow-hidden animate-fade-in">
                        <button
                            onClick={() => handleExport('png')}
                            className="w-full px-4 py-3 text-left text-sm text-primary hover:bg-surfaceHover transition-colors flex items-center gap-3"
                            disabled={isExporting}
                        >
                            <Image size={16} className="text-accent" />
                            Export as PNG
                        </button>

                        <button
                            onClick={() => handleExport('pdf')}
                            className="w-full px-4 py-3 text-left text-sm text-primary hover:bg-surfaceHover transition-colors flex items-center gap-3"
                            disabled={isExporting}
                        >
                            <FileText size={16} className="text-danger" />
                            Export as PDF
                        </button>

                        {data && headers && (
                            <button
                                onClick={() => handleExport('csv')}
                                className="w-full px-4 py-3 text-left text-sm text-primary hover:bg-surfaceHover transition-colors flex items-center gap-3"
                                disabled={isExporting}
                            >
                                <FileText size={16} className="text-success" />
                                Export as CSV
                            </button>
                        )}

                        <div className="h-px bg-white/5 my-1" />

                        <button
                            onClick={() => handleExport('copy')}
                            className="w-full px-4 py-3 text-left text-sm text-primary hover:bg-surfaceHover transition-colors flex items-center gap-3"
                            disabled={isExporting}
                        >
                            <Copy size={16} className="text-secondary" />
                            Copy to Clipboard
                        </button>

                        {typeof navigator.share === 'function' && (
                            <button
                                onClick={() => handleExport('share')}
                                className="w-full px-4 py-3 text-left text-sm text-primary hover:bg-surfaceHover transition-colors flex items-center gap-3"
                                disabled={isExporting}
                            >
                                <Share2 size={16} className="text-warning" />
                                Share
                            </button>
                        )}
                    </div>
                </>
            )}
        </div>
    );
}
