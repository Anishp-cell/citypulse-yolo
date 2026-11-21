import { NavLink, Outlet, useNavigate } from 'react-router-dom';
import { useAuth } from '../../hooks/useAuth';
import {
  LayoutDashboard, Map, AlertTriangle, BarChart3,
  Users, Settings, LogOut, Menu, X, ChevronLeft
} from 'lucide-react';
import { useState } from 'react';
import './AdminLayout.css';

const NAV_ITEMS = [
  { to: '/admin',           icon: LayoutDashboard, label: 'Dashboard',  end: true },
  { to: '/admin/map',       icon: Map,             label: 'Live Map' },
  { to: '/admin/incidents', icon: AlertTriangle,   label: 'Incidents' },
  { to: '/admin/analytics', icon: BarChart3,       label: 'Analytics' },
  { to: '/admin/users',     icon: Users,           label: 'Users' },
  { to: '/admin/settings',  icon: Settings,        label: 'Settings' },
];

export default function AdminLayout() {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className={`admin-layout ${collapsed ? 'sidebar-collapsed' : ''}`}>
      {/* Mobile overlay */}
      {mobileOpen && <div className="sidebar-overlay" onClick={() => setMobileOpen(false)} />}

      {/* Sidebar */}
      <aside className={`admin-sidebar ${mobileOpen ? 'mobile-open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <span className="logo-icon">🚨</span>
            {!collapsed && <span className="logo-text">CityPulse</span>}
          </div>
          <button className="sidebar-collapse-btn desktop-only" onClick={() => setCollapsed(!collapsed)}>
            <ChevronLeft size={18} />
          </button>
          <button className="sidebar-close-btn mobile-only" onClick={() => setMobileOpen(false)}>
            <X size={20} />
          </button>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map(item => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `sidebar-link ${isActive ? 'active' : ''}`}
              onClick={() => setMobileOpen(false)}
            >
              <item.icon size={20} />
              {!collapsed && <span>{item.label}</span>}
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="sidebar-user">
            <div className="user-avatar">
              {(user?.email || 'A')[0].toUpperCase()}
            </div>
            {!collapsed && (
              <div className="user-meta">
                <span className="user-email">{user?.email}</span>
                <span className="user-role">{user?.role}</span>
              </div>
            )}
          </div>
          <button className="sidebar-link logout-link" onClick={handleLogout}>
            <LogOut size={20} />
            {!collapsed && <span>Sign Out</span>}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <div className="admin-main">
        <header className="admin-header">
          <button className="mobile-menu-btn mobile-only" onClick={() => setMobileOpen(true)}>
            <Menu size={22} />
          </button>
          <div className="header-spacer" />
          <div className="header-badge">
            <span className={`role-badge role-${user?.role}`}>{user?.role}</span>
          </div>
        </header>
        <main className="admin-content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
