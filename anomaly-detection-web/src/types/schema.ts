import { z } from 'zod';

// Zod Schema for Health Data Packet
// Ensures incoming data from WebSocket (or Simulator) matches expected format
export const AccelerometerSchema = z.object({
    x: z.number(),
    y: z.number(),
    z: z.number(),
});

export const HealthDataPacketSchema = z.object({
    heartRate: z.number().min(0).max(250),
    hrv: z.number().min(0).max(200),
    stress: z.number().min(0).max(100),
    accelerometer: AccelerometerSchema,
    timestamp: z.number(), // Unix timestamp
    hasAnomaly: z.boolean().optional(), // For labelled data
});

export type HealthDataPacket = z.infer<typeof HealthDataPacketSchema>;

export const validatePacket = (data: any): HealthDataPacket | null => {
    const result = HealthDataPacketSchema.safeParse(data);
    if (result.success) {
        return result.data;
    }
    console.error("Invalid Data Packet:", result.error);
    return null;
};
