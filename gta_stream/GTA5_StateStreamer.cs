// GTA5_StateStreamer.cs — SHVDN Script
// Streams ego vehicle state plus a global top-K nearby world view:
// vehicles, peds, and driving-relevant static objects.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Pipes;
using System.Text;
using System.Threading;
using GTA;
using GTA.Math;
using GTA.Native;

public class GTA5StateStreamer : Script
{
    private const string PIPE_NAME             = "GTA5State";
    private const string CONTROL_PIPE_NAME     = "GTA5Control";
    private const int    TICK_INTERVAL_MS      = 50;
    private const int    MAX_NEARBY_TOTAL      = 32;
    private const int    COMMAND_TIMEOUT_MS    = 250;
    private const int    CONTROL_GROUP         = 0;
    private const int    INPUT_VEH_MOVE_LR     = 59;
    private const int    INPUT_VEH_ACCELERATE  = 71;
    private const int    INPUT_VEH_BRAKE       = 72;
    private const int    INPUT_VEH_DUCK        = 73;
    private const int    INPUT_VEH_HANDBRAKE   = 76;
    private const int    INPUT_VEH_LOOK_BEHIND = 79;
    private const int    INPUT_VEH_HORN        = 86;
    private const float  NEARBY_RADIUS         = 120f;
    private const float  MAX_OBJECT_Z_DELTA    = 20f;
    private const float  ROADSIDE_BAND_METRES  = 25f;

    private const int ENTITY_TYPE_VEHICLE = 1;
    private const int ENTITY_TYPE_PED      = 2;
    private const int ENTITY_TYPE_OBJECT   = 3;

    private NamedPipeServerStream _pipe;
    private StreamWriter          _writer;
    private Thread                _pipeThread;
    private volatile bool         _connected;
    private volatile bool         _serverRunning;
    private NamedPipeServerStream _controlPipe;
    private StreamReader          _controlReader;
    private Thread                _controlThread;
    private volatile bool         _controlConnected;
    private volatile bool         _controlServerRunning;
    private long                  _lastTick;
    private readonly object       _driveLock = new object();
    private DriveCommand          _driveCommand = DriveCommand.Disabled();
    private long                  _lastCommandTick;

    private StreamWriter _log;
    private bool         _logEnabled;
    private long         _frameCount;
    private long         _lastStatTick;
    private long         _lastNeighborWarnTick;

    private static readonly BucketInfo BUCKET_UNKNOWN         = new BucketInfo("unknown", 0);
    private static readonly BucketInfo BUCKET_VEHICLE         = new BucketInfo("vehicle", 1);
    private static readonly BucketInfo BUCKET_PEDESTRIAN      = new BucketInfo("pedestrian", 2);
    private static readonly BucketInfo BUCKET_MOTORCYCLE      = new BucketInfo("motorcycle", 3);
    private static readonly BucketInfo BUCKET_BICYCLE         = new BucketInfo("bicycle", 4);
    private static readonly BucketInfo BUCKET_TRAIN           = new BucketInfo("train", 5);
    private static readonly BucketInfo BUCKET_EMERGENCY       = new BucketInfo("emergency_vehicle", 6);
    private static readonly BucketInfo BUCKET_SERVICE         = new BucketInfo("service_vehicle", 7);
    private static readonly BucketInfo BUCKET_TRAFFIC_LIGHT   = new BucketInfo("traffic_light", 10);
    private static readonly BucketInfo BUCKET_STREET_LIGHT    = new BucketInfo("street_light", 11);
    private static readonly BucketInfo BUCKET_POLE            = new BucketInfo("pole", 12);
    private static readonly BucketInfo BUCKET_SIGN            = new BucketInfo("sign", 13);
    private static readonly BucketInfo BUCKET_BARRIER         = new BucketInfo("barrier", 14);
    private static readonly BucketInfo BUCKET_CURB            = new BucketInfo("curb", 15);
    private static readonly BucketInfo BUCKET_WALL            = new BucketInfo("wall", 16);
    private static readonly BucketInfo BUCKET_TREE            = new BucketInfo("tree", 17);
    private static readonly BucketInfo BUCKET_BUSH            = new BucketInfo("bush", 18);
    private static readonly BucketInfo BUCKET_VEGETATION      = new BucketInfo("vegetation", 19);
    private static readonly BucketInfo BUCKET_BUILDING_EDGE   = new BucketInfo("building_edge", 20);
    private static readonly BucketInfo BUCKET_ROADSIDE_PROP   = new BucketInfo("roadside_prop", 21);

    private static readonly Dictionary<int, BucketInfo> OBJECT_BUCKETS = BuildObjectBuckets();

    private static long NowMs() { return (long)((uint)Environment.TickCount); }

    private sealed class BucketInfo
    {
        public readonly string name;
        public readonly int id;

        public BucketInfo(string name, int id)
        {
            this.name = name;
            this.id = id;
        }
    }

    private struct NearCandidate
    {
        public int handle;
        public int typeId;
        public float dist;

        public NearCandidate(int handle, int typeId, float dist)
        {
            this.handle = handle;
            this.typeId = typeId;
            this.dist = dist;
        }
    }

    private sealed class DriveCommand
    {
        public readonly bool enabled;
        public readonly float steer;
        public readonly float throttle;
        public readonly float brake;
        public readonly bool handbrake;
        public readonly long sentAtMs;

        public DriveCommand(bool enabled, float steer, float throttle, float brake, bool handbrake, long sentAtMs)
        {
            this.enabled = enabled;
            this.steer = Clamp11(steer);
            this.throttle = Clamp01(throttle);
            this.brake = Clamp01(brake);
            this.handbrake = handbrake;
            this.sentAtMs = sentAtMs;
        }

        public static DriveCommand Disabled()
        {
            return new DriveCommand(false, 0f, 0f, 0f, false, 0L);
        }
    }

    public GTA5StateStreamer()
    {
        InitLog();
        Tick += OnTick;
        Aborted += OnAborted;
        StartPipeServer();
        StartControlPipeServer();
    }

    // ── Pipe ───────────────────────────────────────────────────────────────

    private void Log(string level, string msg)
    {
        if (!_logEnabled) return;
        try
        {
            if (_log != null)
            {
                _log.WriteLine("[" + DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss.fff") + "] [" + level + "] " + msg);
                _log.Flush();
            }
        }
        catch { }
    }

    private void InitLog()
    {
        try
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string trimmed = baseDir.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            string dir = string.Equals(Path.GetFileName(trimmed), "scripts", StringComparison.OrdinalIgnoreCase)
                ? trimmed
                : Path.Combine(baseDir, "scripts");
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, "GTA5StateStreamer.log");
            if (File.Exists(path) && new FileInfo(path).Length > 5 * 1024 * 1024)
            {
                string bk = Path.ChangeExtension(path, ".1.log");
                if (File.Exists(bk)) File.Delete(bk);
                File.Move(path, bk);
            }
            _log = new StreamWriter(path, true, Encoding.UTF8);
            _log.AutoFlush = false;
            _logEnabled = true;
        }
        catch { }
    }

    private void StartPipeServer()
    {
        if (_serverRunning) return;
        _serverRunning = true;
        _pipeThread = new Thread(() =>
        {
            while (true)
            {
                try
                {
                    _pipe = new NamedPipeServerStream(
                        PIPE_NAME, PipeDirection.Out, 1,
                        PipeTransmissionMode.Byte, PipeOptions.None);
                    _pipe.WaitForConnection();
                    _writer = new StreamWriter(_pipe, new UTF8Encoding(false));
                    _writer.AutoFlush = true;
                    _connected = true;
                    _frameCount = 0;
                    _logEnabled = true;
                    _serverRunning = false;
                    Log("INFO", "Client connected — streaming started");
                    return;
                }
                catch
                {
                    if (_pipe != null) { try { _pipe.Close(); } catch { } _pipe = null; }
                    Thread.Sleep(2000);
                }
            }
        });
        _pipeThread.IsBackground = true;
        _pipeThread.Start();
    }

    private void StartControlPipeServer()
    {
        if (_controlServerRunning) return;
        _controlServerRunning = true;
        _controlThread = new Thread(() =>
        {
            while (_controlServerRunning)
            {
                try
                {
                    _controlPipe = new NamedPipeServerStream(
                        CONTROL_PIPE_NAME, PipeDirection.In, 1,
                        PipeTransmissionMode.Byte, PipeOptions.None);
                    _controlPipe.WaitForConnection();
                    _controlReader = new StreamReader(_controlPipe, new UTF8Encoding(false));
                    _controlConnected = true;
                    Log("INFO", "Control client connected — awaiting drive commands");

                    string line;
                    while ((line = _controlReader.ReadLine()) != null)
                    {
                        DriveCommand cmd;
                        if (!TryParseDriveCommand(line, out cmd))
                        {
                            Log("WARN", "Ignoring malformed control command: " + line);
                            continue;
                        }

                        lock (_driveLock)
                        {
                            _driveCommand = cmd;
                            _lastCommandTick = NowMs();
                        }
                    }
                }
                catch (Exception ex)
                {
                    Log("WARN", "Control pipe error: " + ex.GetType().Name + " " + ex.Message);
                }
                finally
                {
                    _controlConnected = false;
                    lock (_driveLock)
                    {
                        _driveCommand = DriveCommand.Disabled();
                        _lastCommandTick = 0L;
                    }
                    if (_controlReader != null) { try { _controlReader.Close(); } catch { } _controlReader = null; }
                    if (_controlPipe != null)   { try { _controlPipe.Close();   } catch { } _controlPipe = null; }
                }

                if (!_controlServerRunning) break;
                Thread.Sleep(250);
            }
        });
        _controlThread.IsBackground = true;
        _controlThread.Start();
    }

    private void OnAborted(object sender, EventArgs e)
    {
        Log("INFO", "Script aborted — frames=" + _frameCount);
        _connected = false;
        _serverRunning = false;
        _controlConnected = false;
        _controlServerRunning = false;
        if (_writer != null) { try { _writer.Close(); } catch { } }
        if (_pipe   != null) { try { _pipe.Close();   } catch { } }
        if (_controlReader != null) { try { _controlReader.Close(); } catch { } }
        if (_controlPipe   != null) { try { _controlPipe.Close();   } catch { } }
        if (_log    != null) { try { _log.Close();    } catch { } }
    }

    // ── Tick ───────────────────────────────────────────────────────────────

    private void OnTick(object sender, EventArgs e)
    {
        int playerId = Function.Call<int>(Hash.PLAYER_ID);
        int ped      = Function.Call<int>(Hash.PLAYER_PED_ID);
        if (ped == 0) return;

        bool inVehicle = Function.Call<bool>(Hash.IS_PED_IN_ANY_VEHICLE, ped, false);
        int veh = inVehicle ? Function.Call<int>(Hash.GET_VEHICLE_PED_IS_IN, ped, false) : 0;

        if (inVehicle && veh != 0)
        {
            ApplyDriveCommand();
        }

        if (!_connected) return;
        if (!inVehicle || veh == 0) return;

        long now = NowMs();
        if (now - _lastTick < TICK_INTERVAL_MS) return;
        _lastTick = now;

        string json;
        try { json = BuildJson(now, playerId, ped, veh); }
        catch (Exception ex)
        {
            Log("WARN", "Build error: " + ex.GetType().Name + " " + ex.Message);
            return;
        }

        try
        {
            _writer.WriteLine(json);
            _frameCount++;
            if (now - _lastStatTick >= 5000)
            {
                _lastStatTick = now;
                Log("STAT", "frames=" + _frameCount);
            }
        }
        catch (Exception ex)
        {
            Log("WARN", "Write failed: " + ex.Message);
            _connected = false;
            if (_writer != null) { try { _writer.Close(); } catch { } _writer = null; }
            if (_pipe   != null) { try { _pipe.Close();   } catch { } _pipe   = null; }
            StartPipeServer();
        }
    }

    private void ApplyDriveCommand()
    {
        long now = NowMs();
        DriveCommand cmd;
        bool active;
        long ageMs;
        GetDriveCommandSnapshot(now, out cmd, out active, out ageMs);
        if (!active) return;

        Function.Call(Hash.DISABLE_CONTROL_ACTION, CONTROL_GROUP, INPUT_VEH_MOVE_LR, true);
        Function.Call(Hash.DISABLE_CONTROL_ACTION, CONTROL_GROUP, INPUT_VEH_ACCELERATE, true);
        Function.Call(Hash.DISABLE_CONTROL_ACTION, CONTROL_GROUP, INPUT_VEH_BRAKE, true);
        Function.Call(Hash.DISABLE_CONTROL_ACTION, CONTROL_GROUP, INPUT_VEH_HANDBRAKE, true);

        Function.Call(Hash.SET_CONTROL_NORMAL, CONTROL_GROUP, INPUT_VEH_MOVE_LR, cmd.steer);
        Function.Call(Hash.SET_CONTROL_NORMAL, CONTROL_GROUP, INPUT_VEH_ACCELERATE, cmd.throttle);
        Function.Call(Hash.SET_CONTROL_NORMAL, CONTROL_GROUP, INPUT_VEH_BRAKE, cmd.brake);
        Function.Call(Hash.SET_CONTROL_NORMAL, CONTROL_GROUP, INPUT_VEH_HANDBRAKE, cmd.handbrake ? 1f : 0f);
    }

    private void GetDriveCommandSnapshot(long now, out DriveCommand cmd, out bool active, out long ageMs)
    {
        lock (_driveLock)
        {
            cmd = _driveCommand;
            ageMs = _lastCommandTick <= 0 ? 0 : now - _lastCommandTick;
        }
        active = cmd.enabled && _lastCommandTick > 0 && ageMs <= COMMAND_TIMEOUT_MS;
    }

    private static float ReadControlAxis(int controlId)
    {
        try { return Function.Call<float>(Hash.GET_DISABLED_CONTROL_NORMAL, CONTROL_GROUP, controlId); }
        catch
        {
            try { return Function.Call<float>(Hash.GET_CONTROL_NORMAL, CONTROL_GROUP, controlId); }
            catch { return 0f; }
        }
    }

    private static bool ReadControlPressed(int controlId)
    {
        try { return Function.Call<bool>(Hash.IS_DISABLED_CONTROL_PRESSED, CONTROL_GROUP, controlId); }
        catch
        {
            try { return Function.Call<bool>(Hash.IS_CONTROL_PRESSED, CONTROL_GROUP, controlId); }
            catch { return false; }
        }
    }

    private static float ReadEffectiveControlAxis(int controlId)
    {
        try { return Function.Call<float>(Hash.GET_CONTROL_NORMAL, CONTROL_GROUP, controlId); }
        catch { return 0f; }
    }

    private static bool TryParseDriveCommand(string line, out DriveCommand cmd)
    {
        cmd = DriveCommand.Disabled();
        if (string.IsNullOrWhiteSpace(line)) return false;

        bool enabled = true;
        bool handbrake = false;
        float steer = 0f;
        float throttle = 0f;
        float brake = 0f;
        long sentAtMs = 0L;

        bool hasSteer = TryExtractJsonFloat(line, "steer", out steer);
        bool hasThrottle = TryExtractJsonFloat(line, "throttle", out throttle);
        bool hasBrake = TryExtractJsonFloat(line, "brake", out brake);

        bool hasEnabled = false;
        bool parsedEnabled;
        if (TryExtractJsonBool(line, "enabled", out parsedEnabled))
        {
            enabled = parsedEnabled;
            hasEnabled = true;
        }

        bool parsedHandbrake;
        if (TryExtractJsonBool(line, "handbrake", out parsedHandbrake)) handbrake = parsedHandbrake;

        long parsedSentAt;
        if (TryExtractJsonLong(line, "sent_at_ms", out parsedSentAt)) sentAtMs = parsedSentAt;

        if (!hasSteer && !hasThrottle && !hasBrake && !hasEnabled)
        {
            return false;
        }

        cmd = new DriveCommand(enabled, steer, throttle, brake, handbrake, sentAtMs);
        return true;
    }

    private static bool TryExtractJsonFloat(string json, string key, out float value)
    {
        value = 0f;
        string raw;
        if (!TryExtractJsonToken(json, key, out raw)) return false;
        return float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
    }

    private static bool TryExtractJsonLong(string json, string key, out long value)
    {
        value = 0L;
        string raw;
        if (!TryExtractJsonToken(json, key, out raw)) return false;
        return long.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
    }

    private static bool TryExtractJsonBool(string json, string key, out bool value)
    {
        value = false;
        string raw;
        if (!TryExtractJsonToken(json, key, out raw)) return false;

        raw = raw.Trim();
        if (string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase)) { value = true; return true; }
        if (string.Equals(raw, "false", StringComparison.OrdinalIgnoreCase)) { value = false; return true; }
        if (raw == "1") { value = true; return true; }
        if (raw == "0") { value = false; return true; }
        return false;
    }

    private static bool TryExtractJsonToken(string json, string key, out string raw)
    {
        raw = null;
        string needle = "\"" + key + "\"";
        int keyPos = json.IndexOf(needle, StringComparison.OrdinalIgnoreCase);
        if (keyPos < 0) return false;

        int colonPos = json.IndexOf(':', keyPos + needle.Length);
        if (colonPos < 0) return false;

        int start = colonPos + 1;
        while (start < json.Length && char.IsWhiteSpace(json[start])) start++;
        if (start >= json.Length) return false;

        int end = start;
        char c = json[start];
        if (c == 't' || c == 'T' || c == 'f' || c == 'F')
        {
            while (end < json.Length && char.IsLetter(json[end])) end++;
        }
        else
        {
            while (end < json.Length)
            {
                char ch = json[end];
                if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E')
                {
                    end++;
                    continue;
                }
                break;
            }
        }

        if (end <= start) return false;
        raw = json.Substring(start, end - start);
        return true;
    }

    // ── JSON builder ───────────────────────────────────────────────────────

    private string BuildJson(long now, int playerId, int ped, int veh)
    {
        // Player
        Vector3 pos = Function.Call<Vector3>(Hash.GET_ENTITY_COORDS, ped, true);
        Vector3 vel = Function.Call<Vector3>(Hash.GET_ENTITY_VELOCITY, ped);
        Vector3 rot = Function.Call<Vector3>(Hash.GET_ENTITY_ROTATION, ped, 2);
        int hp      = Function.Call<int>(Hash.GET_ENTITY_HEALTH, ped);
        int hpMax   = Function.Call<int>(Hash.GET_ENTITY_MAX_HEALTH, ped);
        int armor   = Function.Call<int>(Hash.GET_PED_ARMOUR, ped);
        bool dead   = Function.Call<bool>(Hash.IS_ENTITY_DEAD, ped);
        int wanted  = Function.Call<int>(Hash.GET_PLAYER_WANTED_LEVEL, playerId);

        // World
        int clockH  = Function.Call<int>(Hash.GET_CLOCK_HOURS);
        int clockM  = Function.Call<int>(Hash.GET_CLOCK_MINUTES);
        int clockS  = Function.Call<int>(Hash.GET_CLOCK_SECONDS);
        float rain  = Function.Call<float>(Hash.GET_RAIN_LEVEL);
        float wind  = Function.Call<float>(Hash.GET_WIND_SPEED);
        Vector3 wd  = Function.Call<Vector3>(Hash.GET_WIND_DIRECTION);

        // Road — closest node + heading
        float rdHead = 0f, rdDist = 0f;
        Vector3 rdNode = Vector3.Zero;
        try
        {
            var op = new OutputArgument();
            var oh = new OutputArgument();
            Function.Call(Hash.GET_CLOSEST_VEHICLE_NODE_WITH_HEADING,
                pos.X, pos.Y, pos.Z, op, oh, 1, 3, 0);
            rdNode = op.GetResult<Vector3>();
            rdHead = oh.GetResult<float>();
            rdDist = Dist2D(pos, rdNode);
        }
        catch { }

        // Lane — lateral offset, heading delta, lane count, and look-ahead
        float laneOffset = 0f, laneHeadDelta = 0f;
        int   roadLanes  = 0;
        float rd2Head = 0f, rd2Dist = 0f, rdCurve = 0f;
        try
        {
            // Signed lateral offset: project (pos - roadNode) onto the road's
            // right-hand vector so positive = right of centre, negative = left.
            float rdRad = rdHead * (float)(Math.PI / 180.0);
            float rdRightX = (float)Math.Cos(rdRad);
            float rdRightY = (float)(-Math.Sin(rdRad));
            float dx = pos.X - rdNode.X;
            float dy = pos.Y - rdNode.Y;
            laneOffset = dx * rdRightX + dy * rdRightY;

            // Heading delta: how far the vehicle heading deviates from road
            laneHeadDelta = ((vHead - rdHead) % 360f + 360f) % 360f;
            if (laneHeadDelta > 180f) laneHeadDelta -= 360f;

            // Lane count + look-ahead via GET_NTH_CLOSEST_VEHICLE_NODE_WITH_HEADING
            // nthClosest=1 gives us lane count at current position
            var oP1 = new OutputArgument();
            var oH1 = new OutputArgument();
            var oL1 = new OutputArgument();
            Function.Call(Hash.GET_NTH_CLOSEST_VEHICLE_NODE_WITH_HEADING,
                pos.X, pos.Y, pos.Z, 1,
                oP1, oH1, oL1, 1, 3.0f, 0f);
            roadLanes = oL1.GetResult<int>();
            if (roadLanes < 0) roadLanes = 0;

            // nthClosest=2 gives the next road node ahead for curvature
            var oP2 = new OutputArgument();
            var oH2 = new OutputArgument();
            var oL2 = new OutputArgument();
            Function.Call(Hash.GET_NTH_CLOSEST_VEHICLE_NODE_WITH_HEADING,
                pos.X, pos.Y, pos.Z, 2,
                oP2, oH2, oL2, 1, 3.0f, 0f);
            Vector3 rn2 = oP2.GetResult<Vector3>();
            rd2Head = oH2.GetResult<float>();
            rd2Dist = Dist2D(pos, rn2);

            // Curvature: heading change between the two nearest nodes
            rdCurve = ((rd2Head - rdHead) % 360f + 360f) % 360f;
            if (rdCurve > 180f) rdCurve -= 360f;
        }
        catch { }

        // Vehicle — native calls
        float vSpd   = Function.Call<float>(Hash.GET_ENTITY_SPEED, veh);
        float vEngHp = Function.Call<float>(Hash.GET_VEHICLE_ENGINE_HEALTH, veh);
        float vBdHp  = Function.Call<float>(Hash.GET_VEHICLE_BODY_HEALTH, veh);
        bool  vEngOn = Function.Call<bool>(Hash.GET_IS_VEHICLE_ENGINE_RUNNING, veh);
        int   vMod   = Function.Call<int>(Hash.GET_ENTITY_MODEL, veh);
        int   vCls   = Function.Call<int>(Hash.GET_VEHICLE_CLASS, veh);
        float vHead  = Function.Call<float>(Hash.GET_ENTITY_HEADING, veh);
        Vector3 vFwd = Function.Call<Vector3>(Hash.GET_ENTITY_FORWARD_VECTOR, veh);
        Vector3 vAcc = Function.Call<Vector3>(Hash.GET_ENTITY_SPEED_VECTOR, veh, true);

        Vector3 vR = Vector3.Cross(vFwd, Vector3.WorldUp);
        if (vR.Length() < 0.001f) vR = new Vector3(1f, 0f, 0f);
        else vR.Normalize();

        float uThr  = ReadControlAxis(INPUT_VEH_ACCELERATE);
        float uBrk  = ReadControlAxis(INPUT_VEH_BRAKE);
        float uStr  = ReadControlAxis(INPUT_VEH_MOVE_LR);
        bool  uHand = ReadControlPressed(INPUT_VEH_HANDBRAKE);
        bool  uHorn = ReadControlPressed(INPUT_VEH_HORN);
        bool  uLook = ReadControlPressed(INPUT_VEH_LOOK_BEHIND);
        bool  uDuck = ReadControlPressed(INPUT_VEH_DUCK);

        float vThr  = ReadEffectiveControlAxis(INPUT_VEH_ACCELERATE);
        float vBrk  = ReadEffectiveControlAxis(INPUT_VEH_BRAKE);
        float vStr  = ReadEffectiveControlAxis(INPUT_VEH_MOVE_LR);

        DriveCommand cmdSnapshot;
        bool agentActive;
        long agentAgeMs;
        GetDriveCommandSnapshot(now, out cmdSnapshot, out agentActive, out agentAgeMs);

        float pvW = 0f, pvL = 0f, pvH = 0f;
        GetDims(vMod, out pvW, out pvL, out pvH);

        // Vehicle drivetrain state via SHVDN Vehicle properties
        float vRpm = 0f, vTurbo = 0f, vFuel = 0f;
        float vSteerAngle = 0f, vClutch = 0f, vWheelSpd = 0f, vAccel = 0f;
        int   vGear = 0, vMaxG = 0;
        try
        {
            Vehicle vehicle = (Vehicle)Entity.FromHandle(veh);
            if (vehicle != null)
            {
                vRpm        = vehicle.CurrentRPM;
                vGear       = vehicle.CurrentGear;
                vMaxG       = vehicle.HighGear;
                vTurbo      = vehicle.Turbo;
                vSteerAngle = vehicle.SteeringAngle;
                vClutch     = vehicle.Clutch;
                vWheelSpd   = vehicle.WheelSpeed;
                vAccel      = vehicle.Acceleration;
            }
        }
        catch { }

        // Waypoint
        float wpX = 0f, wpY = 0f, wpD = 0f;
        try
        {
            if (Function.Call<bool>(Hash.IS_WAYPOINT_ACTIVE))
            {
                int bl = Function.Call<int>(Hash.GET_FIRST_BLIP_INFO_ID, 8);
                if (bl != 0)
                {
                    Vector3 wp = Function.Call<Vector3>(Hash.GET_BLIP_INFO_ID_COORD, bl);
                    wpX = wp.X;
                    wpY = wp.Y;
                    wpD = Dist2D(pos, wp);
                }
            }
        }
        catch { }

        // Mixed nearby world state
        List<NearCandidate> neighbors = GatherNeighbors(ped, veh, pos);
        int mixedKept = Math.Min(neighbors.Count, MAX_NEARBY_TOTAL);
        var nearVehs    = new List<string>();
        var nearPeds    = new List<string>();
        var nearObjects = new List<string>();
        var nearAll     = new List<string>();

        for (int i = 0; i < neighbors.Count; i++)
        {
            NearCandidate cand = neighbors[i];
            string entry;
            try
            {
                if (cand.typeId == ENTITY_TYPE_VEHICLE)
                {
                    entry = VehJson(cand.handle, i, pos, vel, vFwd, vR, vHead);
                    nearVehs.Add(entry);
                }
                else if (cand.typeId == ENTITY_TYPE_PED)
                {
                    entry = PedJson(cand.handle, i, pos, vel, vFwd, vR, vHead);
                    nearPeds.Add(entry);
                }
                else
                {
                    entry = ObjectJson(cand.handle, i, pos, vel, vFwd, vR, vHead);
                    nearObjects.Add(entry);
                }
                if (i < mixedKept)
                {
                    nearAll.Add(entry);
                }
            }
            catch { }
        }

        var sb = new StringBuilder(16384);
        sb.Append("{");
        sb.Append("\"ts\":" + now + ",");
        sb.Append("\"px\":" + F(pos.X) + ",\"py\":" + F(pos.Y) + ",\"pz\":" + F(pos.Z) + ",");
        sb.Append("\"vx\":" + F(vel.X) + ",\"vy\":" + F(vel.Y) + ",\"vz\":" + F(vel.Z) + ",");
        sb.Append("\"rx\":" + F(rot.X) + ",\"ry\":" + F(rot.Y) + ",\"rz\":" + F(rot.Z) + ",");
        sb.Append("\"hp\":" + hp + ",\"hp_max\":" + hpMax + ",\"armor\":" + armor + ",");
        sb.Append("\"dead\":" + B(dead) + ",\"wanted\":" + wanted + ",");
        sb.Append("\"clock_h\":" + clockH + ",\"clock_m\":" + clockM + ",\"clock_s\":" + clockS + ",");
        sb.Append("\"weather\":0,\"rain\":" + F(rain) + ",");
        sb.Append("\"wind_speed\":" + F(wind) + ",\"wind_x\":" + F(wd.X) + ",\"wind_y\":" + F(wd.Y) + ",");
        sb.Append("\"road_heading\":" + F(rdHead) + ",\"road_dist\":" + F(rdDist) + ",");
        sb.Append("\"lane_offset\":" + F(laneOffset) + ",\"lane_heading_delta\":" + F(laneHeadDelta) + ",");
        sb.Append("\"road_lanes\":" + roadLanes + ",");
        sb.Append("\"road_node2_heading\":" + F(rd2Head) + ",\"road_node2_dist\":" + F(rd2Dist) + ",");
        sb.Append("\"road_curve\":" + F(rdCurve) + ",");
        sb.Append("\"v_speed\":" + F(vSpd) + ",\"v_throttle\":" + F(vThr) + ",\"v_brake\":" + F(vBrk) + ",");
        sb.Append("\"v_steer\":" + F(vStr) + ",\"v_rpm\":" + F(vRpm) + ",");
        sb.Append("\"u_throttle\":" + F(uThr) + ",\"u_brake\":" + F(uBrk) + ",");
        sb.Append("\"u_steer\":" + F(uStr) + ",\"u_handbrake\":" + B(uHand) + ",");
        sb.Append("\"u_horn\":" + B(uHorn) + ",\"u_look_behind\":" + B(uLook) + ",\"u_duck\":" + B(uDuck) + ",");
        sb.Append("\"agent_active\":" + B(agentActive) + ",");
        sb.Append("\"agent_throttle\":" + F(cmdSnapshot.throttle) + ",\"agent_brake\":" + F(cmdSnapshot.brake) + ",");
        sb.Append("\"agent_steer\":" + F(cmdSnapshot.steer) + ",\"agent_handbrake\":" + B(cmdSnapshot.handbrake) + ",");
        sb.Append("\"agent_age_ms\":" + agentAgeMs + ",");
        sb.Append("\"v_gear\":" + vGear + ",\"v_max_gear\":" + vMaxG + ",");
        sb.Append("\"v_rpm\":" + F(vRpm) + ",\"v_steer_angle\":" + F(vSteerAngle) + ",");
        sb.Append("\"v_clutch\":" + F(vClutch) + ",\"v_wheel_speed\":" + F(vWheelSpd) + ",");
        sb.Append("\"v_accel\":" + F(vAccel) + ",");
        sb.Append("\"v_engine_hp\":" + F(vEngHp) + ",\"v_body_hp\":" + F(vBdHp) + ",");
        sb.Append("\"v_engine_on\":" + B(vEngOn) + ",\"v_turbo\":" + F(vTurbo) + ",");
        sb.Append("\"v_fuel\":" + F(vFuel) + ",\"v_model\":" + vMod + ",\"v_class\":" + vCls + ",");
        sb.Append("\"v_ax\":" + F(vAcc.X) + ",\"v_ay\":" + F(vAcc.Y) + ",\"v_az\":" + F(vAcc.Z) + ",");
        sb.Append("\"v_heading\":" + F(vHead) + ",");
        sb.Append("\"v_fwd_x\":" + F(vFwd.X) + ",\"v_fwd_y\":" + F(vFwd.Y) + ",");
        sb.Append("\"v_right_x\":" + F(vR.X) + ",\"v_right_y\":" + F(vR.Y) + ",");
        sb.Append("\"v_dim_w\":" + F(pvW) + ",\"v_dim_l\":" + F(pvL) + ",\"v_dim_h\":" + F(pvH) + ",");
        sb.Append("\"wp_x\":" + F(wpX) + ",\"wp_y\":" + F(wpY) + ",\"wp_dist\":" + F(wpD) + ",");
        sb.Append("\"near_total\":" + neighbors.Count + ",");
        sb.Append("\"near_entities_kept\":" + nearAll.Count + ",");
        sb.Append("\"near_vehs\":[" + string.Join(",", nearVehs) + "],");
        sb.Append("\"near_peds\":[" + string.Join(",", nearPeds) + "],");
        sb.Append("\"near_objects\":[" + string.Join(",", nearObjects) + "],");
        sb.Append("\"near_entities\":[" + string.Join(",", nearAll) + "]");
        sb.Append("}");
        return sb.ToString();
    }

    // ── Candidate gathering ────────────────────────────────────────────────

    private List<NearCandidate> GatherNeighbors(int ped, int veh, Vector3 origin)
    {
        var all = new List<NearCandidate>(96);

        int vehCount = GatherVehicleCandidates(all, veh, origin);
        int pedCount = GatherPedCandidates(all, ped, origin);
        int objCount = GatherObjectCandidates(all, origin);

        all.Sort((a, b) => a.dist.CompareTo(b.dist));

        long now = NowMs();
        if (all.Count == 0 && now - _lastNeighborWarnTick >= 5000)
        {
            _lastNeighborWarnTick = now;
            Log("WARN", "Neighbor gather returned 0 candidates | vehs=" + vehCount + " peds=" + pedCount + " objs=" + objCount);
        }
        return all;
    }

    private int GatherVehicleCandidates(List<NearCandidate> outList, int selfVeh, Vector3 origin)
    {
        int added = 0;
        try
        {
            Vehicle[] allVehicles = World.GetAllVehicles();
            foreach (Vehicle vehicle in allVehicles)
            {
                if (vehicle == null) continue;
                int handle = vehicle.Handle;
                if (handle != 0 && handle != selfVeh && Function.Call<bool>(Hash.DOES_ENTITY_EXIST, handle))
                {
                    Vector3 ep = Function.Call<Vector3>(Hash.GET_ENTITY_COORDS, handle, true);
                    float dist = Dist2D(origin, ep);
                    if (dist < NEARBY_RADIUS)
                    {
                        outList.Add(new NearCandidate(handle, ENTITY_TYPE_VEHICLE, dist));
                        added++;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Log("WARN", "GatherVehicleCandidates failed: " + ex.GetType().Name + " " + ex.Message);
        }
        return added;
    }

    private int GatherPedCandidates(List<NearCandidate> outList, int selfPed, Vector3 origin)
    {
        int added = 0;
        try
        {
            Ped[] allPeds = World.GetAllPeds();
            foreach (Ped ped in allPeds)
            {
                if (ped == null) continue;
                int handle = ped.Handle;
                if (handle != 0 && handle != selfPed && Function.Call<bool>(Hash.DOES_ENTITY_EXIST, handle))
                {
                    Vector3 ep = Function.Call<Vector3>(Hash.GET_ENTITY_COORDS, handle, true);
                    float dist = Dist2D(origin, ep);
                    if (dist < NEARBY_RADIUS)
                    {
                        outList.Add(new NearCandidate(handle, ENTITY_TYPE_PED, dist));
                        added++;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Log("WARN", "GatherPedCandidates failed: " + ex.GetType().Name + " " + ex.Message);
        }
        return added;
    }

    private int GatherObjectCandidates(List<NearCandidate> outList, Vector3 origin)
    {
        int added = 0;
        try
        {
            Prop[] allProps = World.GetAllProps();
            foreach (Prop prop in allProps)
            {
                if (prop == null) continue;
                int handle = prop.Handle;
                if (handle != 0 && Function.Call<bool>(Hash.DOES_ENTITY_EXIST, handle))
                {
                    Vector3 ep = Function.Call<Vector3>(Hash.GET_ENTITY_COORDS, handle, true);
                    float dist = Dist2D(origin, ep);
                    float dz = Math.Abs(ep.Z - origin.Z);
                    if (dist < NEARBY_RADIUS && dz <= MAX_OBJECT_Z_DELTA && ShouldKeepObject(handle, origin, ep, dist))
                    {
                        outList.Add(new NearCandidate(handle, ENTITY_TYPE_OBJECT, dist));
                        added++;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            Log("WARN", "GatherObjectCandidates failed: " + ex.GetType().Name + " " + ex.Message);
        }
        return added;
    }

    // ── Nearby entity JSON builders ───────────────────────────────────────

    private string VehJson(int h, int rank, Vector3 origin, Vector3 pVel, Vector3 fwd, Vector3 right, float pHead)
    {
        Vector3 ep = Function.Call<Vector3>(Hash.GET_ENTITY_COORDS, h, true);
        Vector3 ev = Function.Call<Vector3>(Hash.GET_ENTITY_VELOCITY, h);
        Vector3 d  = new Vector3(ep.X - origin.X, ep.Y - origin.Y, ep.Z - origin.Z);

        float rf = Vector3.Dot(d, fwd);
        float rl = Vector3.Dot(d, right);
        float rz = d.Z;
        float d2 = Dist2D(origin, ep);
        float d3 = d.Length();

        Vector3 dv = new Vector3(ev.X - pVel.X, ev.Y - pVel.Y, ev.Z - pVel.Z);
        float rvf  = Vector3.Dot(dv, fwd);
        float rvl  = Vector3.Dot(dv, right);

        float heading = Function.Call<float>(Hash.GET_ENTITY_HEADING, h);
        float hdiff   = HeadingDiff(heading, pHead);
        Vector3 of    = Function.Call<Vector3>(Hash.GET_ENTITY_FORWARD_VECTOR, h);
        float speed   = Function.Call<float>(Hash.GET_ENTITY_SPEED, h);
        float engHp   = Function.Call<float>(Hash.GET_VEHICLE_ENGINE_HEALTH, h);
        float bodyHp  = Function.Call<float>(Hash.GET_VEHICLE_BODY_HEALTH, h);
        bool stopped  = Function.Call<bool>(Hash.IS_VEHICLE_STOPPED, h);
        int occupants = Function.Call<int>(Hash.GET_VEHICLE_NUMBER_OF_PASSENGERS, h);
        int driver    = Function.Call<int>(Hash.GET_PED_IN_VEHICLE_SEAT, h, -1);
        bool hasDriver = driver != 0 && Function.Call<bool>(Hash.DOES_ENTITY_EXIST, driver);
        int cls       = Function.Call<int>(Hash.GET_VEHICLE_CLASS, h);
        int model     = Function.Call<int>(Hash.GET_ENTITY_MODEL, h);
        bool visible  = EntityVisible(h);
        bool collision = EntityCollisionEnabled(h);

        float dw = 0f, dl = 0f, dh = 0f;
        GetDims(model, out dw, out dl, out dh);

        float ttc = 0f;
        if (rf > 0f && rvf < -0.5f) ttc = -rf / rvf;

        BucketInfo bucket = VehicleBucket(cls);
        bool roadside = IsRoadside(rf, rl, rz, d2);

        var sb = new StringBuilder(512);
        sb.Append("{");
        AppendCommonFields(
            sb, rank, ENTITY_TYPE_VEHICLE, "vehicle", bucket, model,
            ep, rf, rl, rz, d2, d3, ev, speed, rvf, rvl,
            heading, hdiff, of, dw, dl, dh,
            false, collision, visible, roadside);
        sb.Append("\"class\":" + cls + ",");
        sb.Append("\"eng_hp\":" + F(engHp) + ",\"body_hp\":" + F(bodyHp) + ",");
        sb.Append("\"stopped\":" + B(stopped) + ",\"occupants\":" + occupants + ",");
        sb.Append("\"has_driver\":" + B(hasDriver) + ",");
        sb.Append("\"ttc\":" + F(ttc));
        sb.Append("}");
        return sb.ToString();
    }

    private string PedJson(int h, int rank, Vector3 origin, Vector3 pVel, Vector3 fwd, Vector3 right, float pHead)
    {
        Vector3 ep = Function.Call<Vector3>(Hash.GET_ENTITY_COORDS, h, true);
        Vector3 ev = Function.Call<Vector3>(Hash.GET_ENTITY_VELOCITY, h);
        Vector3 d  = new Vector3(ep.X - origin.X, ep.Y - origin.Y, ep.Z - origin.Z);

        float rf = Vector3.Dot(d, fwd);
        float rl = Vector3.Dot(d, right);
        float rz = d.Z;
        float d2 = Dist2D(origin, ep);
        float d3 = d.Length();

        Vector3 dv = new Vector3(ev.X - pVel.X, ev.Y - pVel.Y, ev.Z - pVel.Z);
        float rvf  = Vector3.Dot(dv, fwd);
        float rvl  = Vector3.Dot(dv, right);

        float heading = Function.Call<float>(Hash.GET_ENTITY_HEADING, h);
        float hdiff   = HeadingDiff(heading, pHead);
        Vector3 of    = Function.Call<Vector3>(Hash.GET_ENTITY_FORWARD_VECTOR, h);
        float speed   = Function.Call<float>(Hash.GET_ENTITY_SPEED, h);
        int hp        = Function.Call<int>(Hash.GET_ENTITY_HEALTH, h);
        bool inVeh    = Function.Call<bool>(Hash.IS_PED_IN_ANY_VEHICLE, h, false);
        bool running  = Function.Call<bool>(Hash.IS_PED_RUNNING, h);
        bool walking  = Function.Call<bool>(Hash.IS_PED_WALKING, h);
        int model     = Function.Call<int>(Hash.GET_ENTITY_MODEL, h);
        bool visible  = EntityVisible(h);
        bool collision = EntityCollisionEnabled(h);

        float dw = 0f, dl = 0f, dh = 0f;
        GetDims(model, out dw, out dl, out dh);

        bool crossing = !inVeh && rf > 0f && rf < 30f && Math.Abs(rl) < 4f && Math.Abs(rvl) > 0.3f;

        float ttc = 0f;
        if (rf > 0f && rvf < -0.5f) ttc = -rf / rvf;

        bool roadside = IsRoadside(rf, rl, rz, d2);

        var sb = new StringBuilder(512);
        sb.Append("{");
        AppendCommonFields(
            sb, rank, ENTITY_TYPE_PED, "ped", BUCKET_PEDESTRIAN, model,
            ep, rf, rl, rz, d2, d3, ev, speed, rvf, rvl,
            heading, hdiff, of, dw, dl, dh,
            false, collision, visible, roadside);
        sb.Append("\"hp\":" + hp + ",");
        sb.Append("\"in_veh\":" + B(inVeh) + ",\"running\":" + B(running) + ",\"walking\":" + B(walking) + ",");
        sb.Append("\"crossing\":" + B(crossing) + ",");
        sb.Append("\"ttc\":" + F(ttc));
        sb.Append("}");
        return sb.ToString();
    }

    private string ObjectJson(int h, int rank, Vector3 origin, Vector3 pVel, Vector3 fwd, Vector3 right, float pHead)
    {
        Vector3 ep = Function.Call<Vector3>(Hash.GET_ENTITY_COORDS, h, true);
        Vector3 ev = Function.Call<Vector3>(Hash.GET_ENTITY_VELOCITY, h);
        Vector3 d  = new Vector3(ep.X - origin.X, ep.Y - origin.Y, ep.Z - origin.Z);

        float rf = Vector3.Dot(d, fwd);
        float rl = Vector3.Dot(d, right);
        float rz = d.Z;
        float d2 = Dist2D(origin, ep);
        float d3 = d.Length();

        Vector3 dv = new Vector3(ev.X - pVel.X, ev.Y - pVel.Y, ev.Z - pVel.Z);
        float rvf  = Vector3.Dot(dv, fwd);
        float rvl  = Vector3.Dot(dv, right);

        float heading = 0f;
        try { heading = Function.Call<float>(Hash.GET_ENTITY_HEADING, h); } catch { }
        float hdiff   = HeadingDiff(heading, pHead);
        Vector3 of    = Function.Call<Vector3>(Hash.GET_ENTITY_FORWARD_VECTOR, h);
        float speed   = Function.Call<float>(Hash.GET_ENTITY_SPEED, h);
        int model     = Function.Call<int>(Hash.GET_ENTITY_MODEL, h);
        bool visible  = EntityVisible(h);
        bool collision = EntityCollisionEnabled(h);
        bool attached = EntityAttached(h);

        float dw = 0f, dl = 0f, dh = 0f;
        GetDims(model, out dw, out dl, out dh);
        BucketInfo bucket = ObjectBucket(model, dw, dl, dh, collision, visible);
        bool roadside = IsRoadside(rf, rl, rz, d2);

        var sb = new StringBuilder(512);
        sb.Append("{");
        AppendCommonFields(
            sb, rank, ENTITY_TYPE_OBJECT, "object", bucket, model,
            ep, rf, rl, rz, d2, d3, ev, speed, rvf, rvl,
            heading, hdiff, of, dw, dl, dh,
            true, collision, visible, roadside);
        sb.Append("\"attached\":" + B(attached) + ",");
        sb.Append("\"ttc\":0.0000");
        sb.Append("}");
        return sb.ToString();
    }

    // ── Object selection / semantics ──────────────────────────────────────

    private bool ShouldKeepObject(int handle, Vector3 origin, Vector3 ep, float dist)
    {
        if (EntityAttached(handle)) return false;

        int model = Function.Call<int>(Hash.GET_ENTITY_MODEL, handle);
        float dw = 0f, dl = 0f, dh = 0f;
        GetDims(model, out dw, out dl, out dh);

        float maxDim = Max3(dw, dl, dh);
        if (maxDim < 0.04f) return false;
        if (maxDim > 80f) return false;

        bool collision = EntityCollisionEnabled(handle);
        bool visible   = EntityVisible(handle);
        BucketInfo bucket = ObjectBucket(model, dw, dl, dh, collision, visible);

        if (!collision && !visible && bucket == BUCKET_UNKNOWN) return false;
        if (bucket != BUCKET_UNKNOWN) return true;
        if (!collision && dist > 35f) return false;
        return true;
    }

    private static BucketInfo VehicleBucket(int cls)
    {
        if (cls == 8)  return BUCKET_MOTORCYCLE;
        if (cls == 13) return BUCKET_BICYCLE;
        if (cls == 17) return BUCKET_SERVICE;
        if (cls == 18) return BUCKET_EMERGENCY;
        if (cls == 21) return BUCKET_TRAIN;
        return BUCKET_VEHICLE;
    }

    private static BucketInfo ObjectBucket(int model, float dw, float dl, float dh, bool collision, bool visible)
    {
        BucketInfo mapped;
        if (OBJECT_BUCKETS.TryGetValue(model, out mapped))
        {
            return mapped;
        }

        float footprint = Math.Max(dw, dl);

        if (dh >= 5f && footprint < 1.2f) return BUCKET_POLE;
        if (dh >= 4f && footprint < 2.6f) return BUCKET_STREET_LIGHT;
        if (dh >= 1.4f && dh <= 5f && footprint < 3f && collision) return BUCKET_SIGN;
        if (dh <= 1.5f && footprint >= 1.8f && collision) return BUCKET_BARRIER;
        if (dh <= 0.6f && footprint >= 0.3f && footprint <= 10f && collision) return BUCKET_CURB;
        if ((dw >= 6f || dl >= 6f) && collision) return BUCKET_WALL;
        if (dh >= 2.5f && footprint >= 1.5f && !collision) return BUCKET_TREE;
        if (dh < 2.5f && footprint >= 1.5f && !collision) return BUCKET_BUSH;
        if (dh < 1.5f && footprint >= 0.25f && !collision) return BUCKET_VEGETATION;
        if (dw > 8f || dl > 8f || dh > 8f) return BUCKET_BUILDING_EDGE;
        if (collision || visible) return BUCKET_ROADSIDE_PROP;
        return BUCKET_UNKNOWN;
    }

    private static Dictionary<int, BucketInfo> BuildObjectBuckets()
    {
        var map = new Dictionary<int, BucketInfo>();

        AddBucket(map, BUCKET_TRAFFIC_LIGHT, new string[] {
            "prop_traffic_01a", "prop_traffic_01b", "prop_traffic_01d",
            "prop_traffic_02a", "prop_traffic_02b", "prop_traffic_03a",
            "prop_traffic_03b"
        });
        AddBucket(map, BUCKET_STREET_LIGHT, new string[] {
            "prop_streetlight_01", "prop_streetlight_01b", "prop_streetlight_01d",
            "prop_streetlight_02", "prop_streetlight_03d", "prop_docklight_01"
        });
        AddBucket(map, BUCKET_POLE, new string[] {
            "prop_pole_01a", "prop_pole_02a", "prop_telegraph_01a",
            "prop_telegraph_01b", "prop_telegraph_01c"
        });
        AddBucket(map, BUCKET_SIGN, new string[] {
            "prop_sign_road_01a", "prop_sign_road_03e", "prop_sign_road_04a",
            "prop_sign_road_06a", "prop_sign_road_restriction_10",
            "prop_sign_road_restriction_20"
        });
        AddBucket(map, BUCKET_BARRIER, new string[] {
            "prop_barrier_work05", "prop_barrier_work06a", "prop_barriercrash_01",
            "prop_mp_barrier_01", "prop_mp_barrier_02b", "prop_roadcone02a"
        });
        AddBucket(map, BUCKET_TREE, new string[] {
            "prop_tree_birch_01", "prop_tree_lficus_02", "prop_tree_maple_03",
            "prop_tree_jacada_01", "prop_palm_fan_03_a", "prop_palm_med_01a"
        });
        AddBucket(map, BUCKET_BUSH, new string[] {
            "prop_bush_lrg_01", "prop_bush_med_03", "prop_bush_lrg_04c",
            "prop_plant_group_03"
        });
        AddBucket(map, BUCKET_VEGETATION, new string[] {
            "prop_grass_dry_02", "prop_grass_long_01", "prop_veg_crop_03_cab",
            "prop_plant_fern_02a"
        });
        AddBucket(map, BUCKET_WALL, new string[] {
            "prop_wallbrick_01", "prop_walllight_02a", "prop_fncwall_03c",
            "prop_fnclink_03e"
        });

        return map;
    }

    private static void AddBucket(Dictionary<int, BucketInfo> map, BucketInfo bucket, string[] modelNames)
    {
        foreach (string name in modelNames)
        {
            map[Joaat(name)] = bucket;
        }
    }

    private static int Joaat(string value)
    {
        uint hash = 0;
        for (int i = 0; i < value.Length; i++)
        {
            char c = value[i];
            if (c >= 'A' && c <= 'Z') c = (char)(c + 32);
            hash += c;
            hash += (hash << 10);
            hash ^= (hash >> 6);
        }
        hash += (hash << 3);
        hash ^= (hash >> 11);
        hash += (hash << 15);
        return unchecked((int)hash);
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private static void AppendCommonFields(
        StringBuilder sb,
        int rank,
        int typeId,
        string entityType,
        BucketInfo bucket,
        int model,
        Vector3 ep,
        float rf,
        float rl,
        float rz,
        float d2,
        float d3,
        Vector3 ev,
        float speed,
        float rvf,
        float rvl,
        float heading,
        float hdiff,
        Vector3 of,
        float dw,
        float dl,
        float dh,
        bool isStatic,
        bool collision,
        bool visible,
        bool roadside)
    {
        sb.Append("\"rank\":" + rank + ",");
        sb.Append("\"type_id\":" + typeId + ",\"entity_type\":\"" + entityType + "\",");
        sb.Append("\"bucket_id\":" + bucket.id + ",\"semantic_bucket\":\"" + bucket.name + "\",");
        sb.Append("\"model_hash\":" + model + ",\"hash\":" + model + ",");
        sb.Append("\"x\":" + F(ep.X) + ",\"y\":" + F(ep.Y) + ",\"z\":" + F(ep.Z) + ",");
        sb.Append("\"rel_fwd\":" + F(rf) + ",\"rel_lat\":" + F(rl) + ",\"rel_z\":" + F(rz) + ",");
        sb.Append("\"dist\":" + F(d2) + ",\"dist3d\":" + F(d3) + ",");
        sb.Append("\"speed\":" + F(speed) + ",");
        sb.Append("\"vx\":" + F(ev.X) + ",\"vy\":" + F(ev.Y) + ",\"vz\":" + F(ev.Z) + ",");
        sb.Append("\"rel_v_fwd\":" + F(rvf) + ",\"rel_v_lat\":" + F(rvl) + ",");
        sb.Append("\"heading\":" + F(heading) + ",\"hdiff\":" + F(hdiff) + ",");
        sb.Append("\"fwd_x\":" + F(of.X) + ",\"fwd_y\":" + F(of.Y) + ",");
        sb.Append("\"dim_w\":" + F(dw) + ",\"dim_l\":" + F(dl) + ",\"dim_h\":" + F(dh) + ",");
        sb.Append("\"is_static\":" + B(isStatic) + ",");
        sb.Append("\"has_collision\":" + B(collision) + ",\"is_visible\":" + B(visible) + ",");
        sb.Append("\"is_on_roadside\":" + B(roadside) + ",");
    }

    private static float Dist2D(Vector3 a, Vector3 b)
    {
        float dx = a.X - b.X;
        float dy = a.Y - b.Y;
        return (float)Math.Sqrt(dx * dx + dy * dy);
    }

    private static float HeadingDiff(float entityHeading, float playerHeading)
    {
        float hd = ((entityHeading - playerHeading) % 360f + 360f) % 360f;
        if (hd > 180f) hd -= 360f;
        return hd;
    }

    private static bool IsRoadside(float relFwd, float relLat, float relZ, float dist)
    {
        return dist <= ROADSIDE_BAND_METRES && Math.Abs(relZ) <= 10f && Math.Abs(relLat) <= 20f;
    }

    private static float Max3(float a, float b, float c)
    {
        return Math.Max(a, Math.Max(b, c));
    }

    private static void GetDims(int model, out float w, out float l, out float h)
    {
        w = 0f;
        l = 0f;
        h = 0f;
        try
        {
            var oMin = new OutputArgument();
            var oMax = new OutputArgument();
            Function.Call(Hash.GET_MODEL_DIMENSIONS, model, oMin, oMax);
            Vector3 mn = oMin.GetResult<Vector3>();
            Vector3 mx = oMax.GetResult<Vector3>();
            w = mx.X - mn.X;
            l = mx.Y - mn.Y;
            h = mx.Z - mn.Z;
        }
        catch { }
    }

    private static bool EntityCollisionEnabled(int handle)
    {
        try { return !Function.Call<bool>(Hash.GET_ENTITY_COLLISION_DISABLED, handle); }
        catch { return true; }
    }

    private static bool EntityVisible(int handle)
    {
        try { return Function.Call<bool>(Hash.IS_ENTITY_VISIBLE, handle); }
        catch { return true; }
    }

    private static bool EntityAttached(int handle)
    {
        try { return Function.Call<bool>(Hash.IS_ENTITY_ATTACHED, handle); }
        catch { return false; }
    }

    private static string F(float v)
    {
        if (float.IsNaN(v) || float.IsInfinity(v)) return "0";
        return v.ToString("F4");
    }

    private static float Clamp01(float v)
    {
        if (v < 0f) return 0f;
        if (v > 1f) return 1f;
        return v;
    }

    private static float Clamp11(float v)
    {
        if (v < -1f) return -1f;
        if (v > 1f) return 1f;
        return v;
    }

    private static string B(bool v)
    {
        return v ? "true" : "false";
    }
}
