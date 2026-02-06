import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, BarChart3, Users, Award } from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';
import { Button } from '@/components/ui/Button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card';
import { getModelStats, getRoleStats, getModelRoleStats } from '@/lib/api';
import type { ModelStats, RoleStats, ModelRoleStats } from '@/lib/types';
import { ROLE_NAMES } from '@/lib/utils';

const COLORS = ['#6366f1', '#ec4899', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

export function Stats() {
  const navigate = useNavigate();
  const [modelStats, setModelStats] = useState<ModelStats[]>([]);
  const [roleStats, setRoleStats] = useState<RoleStats[]>([]);
  const [modelRoleStats, setModelRoleStats] = useState<ModelRoleStats[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const [models, roles, modelRoles] = await Promise.all([
        getModelStats(),
        getRoleStats(),
        getModelRoleStats(),
      ]);
      setModelStats(models);
      setRoleStats(roles);
      setModelRoleStats(modelRoles);
    } catch (error) {
      console.error('Failed to load stats:', error);
    } finally {
      setLoading(false);
    }
  };

  // Prepare data for charts
  const modelChartData = modelStats.map((s) => ({
    name: s.model,
    胜率: Math.round(s.win_rate * 100),
    游戏数: s.games_played,
  }));

  const roleChartData = roleStats.map((s) => ({
    name: ROLE_NAMES[s.role as keyof typeof ROLE_NAMES] || s.role,
    胜率: Math.round(s.win_rate * 100),
    游戏数: s.games_played,
  }));

  const gamePieData = modelStats.map((s, i) => ({
    name: s.model,
    value: s.games_played,
    color: COLORS[i % COLORS.length],
  }));

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-muted-foreground">加载中...</div>
      </div>
    );
  }

  const hasData = modelStats.length > 0 || roleStats.length > 0;

  return (
    <div className="container mx-auto py-6 px-4 max-w-6xl">
      {/* Header */}
      <div className="flex items-center gap-4 mb-8">
        <Button variant="ghost" onClick={() => navigate('/')}>
          <ArrowLeft className="mr-2 h-4 w-4" />
          返回
        </Button>
        <h1 className="text-2xl font-bold">统计数据</h1>
      </div>

      {!hasData ? (
        <Card>
          <CardContent className="py-12 text-center text-muted-foreground">
            <BarChart3 className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p>暂无统计数据</p>
            <p className="text-sm">完成一些游戏后这里会显示统计信息</p>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-8">
          {/* Model Win Rate */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Award className="h-5 w-5" />
                模型胜率
              </CardTitle>
            </CardHeader>
            <CardContent>
              {modelChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={modelChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis unit="%" />
                    <Tooltip />
                    <Bar dataKey="胜率" fill="#6366f1" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-center text-muted-foreground py-8">暂无数据</p>
              )}
            </CardContent>
          </Card>

          {/* Role Win Rate */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Users className="h-5 w-5" />
                角色胜率
              </CardTitle>
            </CardHeader>
            <CardContent>
              {roleChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={roleChartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis unit="%" />
                    <Tooltip />
                    <Bar dataKey="胜率" fill="#10b981" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-center text-muted-foreground py-8">暂无数据</p>
              )}
            </CardContent>
          </Card>

          {/* Games Distribution */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                游戏分布
              </CardTitle>
            </CardHeader>
            <CardContent>
              {gamePieData.length > 0 && gamePieData.some((d) => d.value > 0) ? (
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={gamePieData}
                      dataKey="value"
                      nameKey="name"
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      label={({ name, value }) => `${name}: ${value}`}
                    >
                      {gamePieData.map((entry, index) => (
                        <Cell key={index} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              ) : (
                <p className="text-center text-muted-foreground py-8">暂无数据</p>
              )}
            </CardContent>
          </Card>

          {/* Model-Role Stats Table */}
          {modelRoleStats.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>模型-角色详细统计</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2 px-4">模型</th>
                        <th className="text-left py-2 px-4">角色</th>
                        <th className="text-right py-2 px-4">游戏数</th>
                        <th className="text-right py-2 px-4">胜利</th>
                        <th className="text-right py-2 px-4">胜率</th>
                      </tr>
                    </thead>
                    <tbody>
                      {modelRoleStats.map((stat, index) => (
                        <tr key={index} className="border-b last:border-0">
                          <td className="py-2 px-4">{stat.model}</td>
                          <td className="py-2 px-4">
                            {ROLE_NAMES[stat.role as keyof typeof ROLE_NAMES] || stat.role}
                          </td>
                          <td className="text-right py-2 px-4">{stat.games_played}</td>
                          <td className="text-right py-2 px-4">{stat.wins}</td>
                          <td className="text-right py-2 px-4">
                            {Math.round(stat.win_rate * 100)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
