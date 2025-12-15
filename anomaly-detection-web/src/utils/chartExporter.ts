import html2canvas from 'html2canvas';
import jsPDF from 'jspdf';

export class ChartExporter {
    /**
     * Export chart as PNG image
     */
    static async exportAsPNG(elementId: string, filename: string = 'chart.png') {
        const element = document.getElementById(elementId);
        if (!element) {
            throw new Error(`Element with id "${elementId}" not found`);
        }

        const canvas = await html2canvas(element, {
            backgroundColor: '#1C1C1E',
            scale: 2, // Higher quality
        });

        const link = document.createElement('a');
        link.download = filename;
        link.href = canvas.toDataURL('image/png');
        link.click();
    }

    /**
     * Export chart as PDF
     */
    static async exportAsPDF(elementId: string, filename: string = 'chart.pdf') {
        const element = document.getElementById(elementId);
        if (!element) {
            throw new Error(`Element with id "${elementId}" not found`);
        }

        const canvas = await html2canvas(element, {
            backgroundColor: '#1C1C1E',
            scale: 2,
        });

        const imgData = canvas.toDataURL('image/png');
        const pdf = new jsPDF({
            orientation: canvas.width > canvas.height ? 'landscape' : 'portrait',
            unit: 'px',
            format: [canvas.width, canvas.height],
        });

        pdf.addImage(imgData, 'PNG', 0, 0, canvas.width, canvas.height);
        pdf.save(filename);
    }

    /**
     * Export data as CSV
     */
    static exportAsCSV(
        data: any[],
        headers: string[],
        filename: string = 'data.csv'
    ) {
        const csvContent = [
            headers.join(','),
            ...data.map(row =>
                headers.map(header => {
                    const value = row[header];
                    // Escape commas and quotes
                    if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                        return `"${value.replace(/"/g, '""')}"`;
                    }
                    return value;
                }).join(',')
            ),
        ].join('\n');

        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = filename;
        link.click();
    }

    /**
     * Share chart as image (Web Share API)
     */
    static async shareChart(elementId: string, title: string = 'Health Chart') {
        if (!navigator.share) {
            throw new Error('Web Share API not supported');
        }

        const element = document.getElementById(elementId);
        if (!element) {
            throw new Error(`Element with id "${elementId}" not found`);
        }

        const canvas = await html2canvas(element, {
            backgroundColor: '#1C1C1E',
            scale: 2,
        });

        canvas.toBlob(async (blob) => {
            if (!blob) return;

            const file = new File([blob], 'chart.png', { type: 'image/png' });

            try {
                await navigator.share({
                    title,
                    text: 'Check out my health data',
                    files: [file],
                });
            } catch (error) {
                console.error('Error sharing:', error);
            }
        });
    }

    /**
     * Copy chart to clipboard
     */
    static async copyToClipboard(elementId: string) {
        const element = document.getElementById(elementId);
        if (!element) {
            throw new Error(`Element with id "${elementId}" not found`);
        }

        const canvas = await html2canvas(element, {
            backgroundColor: '#1C1C1E',
            scale: 2,
        });

        canvas.toBlob(async (blob) => {
            if (!blob) return;

            try {
                await navigator.clipboard.write([
                    new ClipboardItem({ 'image/png': blob }),
                ]);
            } catch (error) {
                console.error('Error copying to clipboard:', error);
            }
        });
    }
}
